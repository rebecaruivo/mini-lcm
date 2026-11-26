-- Data de lançamento do produto (quando a fonte informa), usada na avaliação por data
-- para saber se o produto já existia num corte sem olhar o comportamento das pessoas.
-- Rode depois do 001: psql "$DATABASE_URL" -f sql/002_lancamento.sql
alter table itens add column if not exists lancamento date;
