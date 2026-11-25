-- mini-lcm: esquema do banco (Postgres + pgvector).
-- Rode uma vez: psql "$DATABASE_URL" -f sql/001_schema.sql
-- (no Supabase, também dá para colar no SQL Editor)
--
-- A fase 1 usa categorias, itens, embeddings_itens, avaliacoes e custos.
-- usuarios e eventos já existem para a fase 2, para o modelo de dados não
-- precisar mudar de forma quando a personalização entrar.

create extension if not exists vector;

-- ───────────── Catálogo ─────────────

create table if not exists categorias (
  slug text primary key,
  nome text not null
);

create table if not exists itens (
  id text primary key,                         -- id na fonte (Amazon: parent_asin)
  fonte text not null default 'amazon',
  categoria text not null references categorias (slug),
  titulo text not null,
  preco numeric(12, 2) check (preco is null or preco >= 0),
  moeda text,
  condicao text not null check (condicao in ('novo', 'usado', 'desconhecida')),
  vendedor_id text,                            -- Amazon: nome da loja
  link text,
  foto_url text,                               -- só a URL: a foto não é copiada para cá
  atributos jsonb not null default '{}'::jsonb,  -- campos que variam por categoria
  chave_duplicata text not null,               -- vendedor + título normalizado
  coletado_em timestamptz not null default now(),
  atualizado_em timestamptz not null default now()
);

create index if not exists itens_categoria_preco on itens (categoria, preco);
create index if not exists itens_chave_duplicata on itens (chave_duplicata);

-- Embeddings ficam fora de "itens" para podermos trocar de modelo sem perder dados:
-- cada (item, modelo, canal) tem seu vetor. A coluna "vetor" não fixa o número de
-- dimensões, porque modelos diferentes têm tamanhos diferentes. O índice HNSW, que
-- exige tamanho fixo, é criado por modelo e canal (ver garantir_indice no código):
--   create index ... using hnsw ((vetor::vector(512)) vector_cosine_ops)
--   where modelo = '...' and canal = '...';
create table if not exists embeddings_itens (
  item_id text not null references itens (id) on delete cascade,
  modelo text not null,
  canal text not null check (canal in ('texto', 'imagem')),
  dimensoes int not null check (dimensoes > 0),
  vetor vector not null,
  criado_em timestamptz not null default now(),
  primary key (item_id, modelo, canal),
  check (vector_dims(vetor) = dimensoes)
);

-- ───────────── Avaliações reais (Amazon) ─────────────
-- Interações de usuários reais e anônimos, como vieram da fonte. Na fase 2 elas
-- alimentam o perfil de cada usuário. Ficam separadas de "eventos" porque não
-- foram geradas pelo nosso sistema: são o histórico que já existia.

create table if not exists avaliacoes (
  fonte text not null default 'amazon',
  usuario_externo text not null,               -- id anônimo do usuário na fonte
  item_id text not null references itens (id) on delete cascade,
  nota real not null check (nota between 1 and 5),
  momento timestamptz not null,
  compra_verificada boolean not null default false,
  primary key (fonte, usuario_externo, item_id, momento)
);

create index if not exists avaliacoes_item on avaliacoes (item_id);
create index if not exists avaliacoes_usuario_momento on avaliacoes (usuario_externo, momento);

-- ───────────── Custos ─────────────

create table if not exists custos (
  id bigint generated always as identity primary key,
  etapa text not null,
  modelo text not null,
  itens int not null default 0,
  chamadas int not null default 0,
  segundos double precision not null default 0,
  tokens int not null default 0,
  custo_usd numeric(14, 8) not null default 0,
  falhas int not null default 0,
  criado_em timestamptz not null default now()
);

-- ───────────── Fase 2: usuários e eventos ─────────────

create table if not exists usuarios (
  id uuid primary key default gen_random_uuid(),
  nome text not null unique,
  tipo text not null check (tipo in ('persona', 'real')),  -- persona sintética ou pessoa de verdade
  criado_em timestamptz not null default now()
);

create table if not exists eventos (
  id bigint generated always as identity primary key,
  usuario_id uuid not null references usuarios (id),
  tipo text not null check (tipo in ('busca', 'visualizacao', 'clique', 'salvar', 'descartar', 'comprar')),
  item_id text references itens (id),           -- vazio em eventos de busca
  consulta text,                                -- preenchido em eventos de busca
  posicao int,                                  -- em que posição o item aparecia quando houve a ação
  criado_em timestamptz not null default now(),
  check ((tipo = 'busca') = (consulta is not null))
);

create index if not exists eventos_usuario_data on eventos (usuario_id, criado_em);

-- Eventos são um histórico: só entram, nunca mudam nem somem.
-- O perfil do usuário é sempre recalculado a partir deles.
create or replace function bloquear_alteracao_eventos() returns trigger
language plpgsql as $$
begin
  raise exception 'eventos é append-only: % não é permitido', tg_op;
end;
$$;

drop trigger if exists eventos_append_only on eventos;
create trigger eventos_append_only
  before update or delete on eventos
  for each row execute function bloquear_alteracao_eventos();
