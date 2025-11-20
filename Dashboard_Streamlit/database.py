# database.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit_authenticator as stauth  # Necessário para gerar hash na criação do admin


@st.cache_resource
def get_connection():
    """Cria e gerencia a conexão com o banco de dados usando o SQLAlchemy."""
    try:
        creds = st.secrets.database
        db_uri = f"postgresql://{creds.user}:{creds.password}@{creds.host}:{creds.port}/{creds.dbname}"
        engine = create_engine(db_uri)
        return engine
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None


def init_users_db(engine):
    """Cria a tabela users e insere o admin padrão se não existir."""
    try:
        with engine.connect() as connection:
            # 1. Criar a tabela users se não existir
            create_table_query = """
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100),
                password_hash TEXT,
                role VARCHAR(20) DEFAULT 'admin',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            connection.execute(text(create_table_query))

            # 2. Verificar se existe algum usuário cadastrado
            check_query = "SELECT COUNT(*) FROM users"
            result = connection.execute(text(check_query)).fetchone()

            # 3. Se a tabela estiver vazia, criar o Admin Padrão
            if result[0] == 0:
                print("Tabela users vazia. Criando admin padrão...")
                default_pass = "admin456"
                # Gera o hash da senha
                hashed_pass = stauth.Hasher([default_pass]).generate()[0]

                insert_query = text(
                    """
                    INSERT INTO users (username, name, password_hash, role)
                    VALUES (:u, :n, :p, :r)
                """
                )
                connection.execute(
                    insert_query,
                    {
                        "u": "admin",
                        "n": "Administrador",
                        "p": hashed_pass,
                        "r": "admin",
                    },
                )
                connection.commit()
                print("Admin padrão criado com sucesso.")
            else:
                connection.commit()

    except Exception as e:
        st.error(f"Erro ao inicializar tabela 'users': {e}")


def get_users_credentials(engine):
    """Busca todos os usuários no banco e retorna no formato exigido pelo Authenticator."""
    try:
        query = "SELECT username, name, password_hash FROM users"
        df = pd.read_sql(query, con=engine)

        credentials = {"usernames": {}}

        for _, row in df.iterrows():
            credentials["usernames"][row["username"]] = {
                "name": row["name"],
                "password": row["password_hash"],
            }

        return credentials
    except Exception as e:
        st.error(f"Erro ao buscar usuários: {e}")
        return {"usernames": {}}


def init_db(engine):
    """Cria a tabela de submissões se ela não existir."""
    table_name = "submissions"

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        user_submitting VARCHAR(50),
        
        -- Dados do formulário
        curso VARCHAR(255),
        semestre_ingresso FLOAT,
        identificacao_curso VARCHAR(100),
        tipo_transporte VARCHAR(50),
        propriedade_transporte VARCHAR(100),
        barreira_transporte VARCHAR(100),
        mora_na_cidade VARCHAR(100),
        tempo_deslocamento INT,
        acessibilidade_campus VARCHAR(50),
        preconceito_cor BOOLEAN,
        preconceito_financeiro BOOLEAN,
        preconceito_aparencia BOOLEAN,
        preconceito_deficiencia BOOLEAN,
        preconceito_aprendizado BOOLEAN,
        preconceito_genero BOOLEAN,
        preconceito_curso BOOLEAN,
        preconceito_idade BOOLEAN,
        preconceito_nao BOOLEAN,
        tempo_estudo VARCHAR(255),
        situacao_trabalho VARCHAR(255),
        horarios_trabalho VARCHAR(255),
        estado_civil BOOLEAN,
        tem_filhos BOOLEAN,
        qtd_filhos INT,
        contribuicao_financeira VARCHAR(255),
        
        -- Respostas abertas
        dificuldades_permanencia TEXT,
        resposta_completa_evasao TEXT,
        pensou_evasao_real INT,

        -- Resultados da predição
        prediction_label INT,
        prediction_score FLOAT,
        
        -- NOVAS COLUNAS PARA DETALHES
        fatores_risco TEXT,
        acoes_recomendadas TEXT,
        categoria_risco VARCHAR(20)
    );
    """

    try:
        with engine.connect() as connection:
            connection.execute(text(create_table_query))
            connection.commit()
            print("Tabela 'submissions' criada/verificada com sucesso!")
    except Exception as e:
        st.error(f"Erro ao inicializar a tabela '{table_name}': {e}")


def save_submission(engine, data: pd.DataFrame):
    """Salva um DataFrame na tabela 'submissions'."""
    try:
        data.to_sql("submissions", con=engine, if_exists="append", index=False)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar dados: {e}")
        return False


def load_data(engine):
    """Carrega todos os dados da tabela 'submissions'."""
    try:
        return pd.read_sql(
            "SELECT * FROM submissions ORDER BY timestamp DESC", con=engine
        )
    except Exception as e:
        st.warning(f"Erro ao carregar dados ou tabela vazia: {e}")
        return pd.DataFrame()


def get_stats(engine):
    """Retorna estatísticas básicas dos dados."""
    try:
        with engine.connect() as connection:
            total_query = "SELECT COUNT(*) as total FROM submissions"
            alto_risco_query = "SELECT COUNT(*) as alto_risco FROM submissions WHERE prediction_label = 1"

            total_result = connection.execute(text(total_query)).fetchone()
            alto_risco_result = connection.execute(text(alto_risco_query)).fetchone()

            total = total_result[0] if total_result else 0
            alto_risco = alto_risco_result[0] if alto_risco_result else 0

            return {
                "total_submissions": total,
                "alto_risco_count": alto_risco,
                "baixo_risco_count": total - alto_risco,
                "percentual_alto_risco": (alto_risco / total * 100) if total > 0 else 0,
            }
    except Exception as e:
        st.error(f"Erro ao calcular estatísticas: {e}")
        return {
            "total_submissions": 0,
            "alto_risco_count": 0,
            "baixo_risco_count": 0,
            "percentual_alto_risco": 0,
        }
