import os
import io
import sys
import traceback
import time
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from google.api_core import exceptions as core_exceptions

matplotlib.use("Agg")

def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
        .stApp { background: radial-gradient(circle, rgba(17,22,30,1) 0%, rgba(11,14,20,1) 100%); color: #E5E9F0; font-family: 'Roboto', sans-serif; }
        .app-title { font-family: 'Orbitron', sans-serif; color: #88C0D0; text-shadow: 0 0 12px rgba(136, 192, 208, 0.6); font-size: 2.8rem; margin-bottom: 0.5rem; }
        .app-sub { font-family: 'Roboto', sans-serif; color: #B48EAD; font-size: 1.1rem; margin-bottom: 2rem; font-weight: 300; }
        [data-testid="stSidebar"], [data-testid="stChatMessage"], [data-testid="stFileUploader"], .stExpander, [data-testid="stTextInput"] { background: rgba(46, 52, 64, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(76, 86, 106, 0.4); border-radius: 12px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2); margin-bottom: 15px; }
        .stChatMessage:has([data-testid="chat-avatar-user"]) { background-color: rgba(94, 129, 172, 0.3); border-left: 5px solid #88C0D0; }
        .stChatMessage:has([data-testid="chat-avatar-assistant"]) { background-color: rgba(67, 76, 94, 0.3); border-left: 5px solid #B48EAD; }
    </style>
    """, unsafe_allow_html=True)

class PandasAgent:
    def __init__(self, df: pd.DataFrame, llm: genai.GenerativeModel, max_retries: int = 2):
        self.df, self.llm, self.max_retries = df, llm, max_retries
        self.df_context = self._build_dataframe_context()

    def _build_dataframe_context(self) -> str:
        buffer = io.StringIO(); self.df.info(buf=buffer, verbose=False); mem_usage = buffer.getvalue().split('\n')[-2]
        return f"O DataFrame 'df' tem {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.\nNomes das colunas e tipos de dados: {self.df.dtypes.to_dict()}\nUso de memória: {mem_usage}\nPrimeiras 5 linhas:\n{self.df.head().to_string()}"

    def _get_router_prompt(self, user_query: str) -> str:
        return f"""
        Sua tarefa é classificar a pergunta de um usuário em UMA das três categorias abaixo. Responda APENAS com o nome da categoria.
        CATEGORIAS:
        1. `dataframe_query`: A pergunta é sobre os dados no arquivo CSV.
        2. `agent_capabilities_query`: A pergunta é sobre o que você, o agente, pode fazer.
        3. `general_knowledge`: A pergunta é um conhecimento geral.
        EXEMPLOS:
        - Pergunta: 'qual a média de valor FOB por mês?' -> Resposta: dataframe_query
        - Pergunta: 'o que vc pode fazer?' -> Resposta: agent_capabilities_query
        - Pergunta: 'quem descobriu o brasil?' -> Resposta: general_knowledge
        ---
        Pergunta a ser classificada: '{user_query}'
        Resposta:
        """

    def _get_codegen_prompt(self, user_query: str, chat_history: list, error_context: dict = None) -> str:
        history_str = "\n".join([f"- {m['role']}: {m.get('content', '')}" for m in chat_history[-4:]])
        error_section = ""
        if error_context: error_section = f"O CÓDIGO ANTERIOR FALHOU...\n<codigo_anterior>\n{error_context['code']}\n</codigo_anterior>\n<mensagem_erro>\n{error_context['error']}\n</mensagem_erro>"
        return f"Você é um expert em análise de dados com Python. Use o contexto da conversa e a nova pergunta para gerar um código.\n\n<contexto_dataframe>\n{self.df_context}\n</contexto_dataframe>\n\n<historico_conversa>\n{history_str}</historico_conversa>\n\n<pergunta_usuario_atual>\n{user_query}\n</pergunta_usuario_atual>\n\n{error_section}\n\nREGRAS:\n1. Use o histórico...\n2. Gere APENAS o código Python.\n3. Use `print()` para saídas de texto."

    def _get_summarizer_prompt(self, user_query: str, result: str) -> str:
        return (f"Como um assistente de dados, explique o seguinte resultado de forma clara. A pergunta foi: '{user_query}'.\n\nResultado a ser explicado:\n'{result}'")

    def _get_capability_examples(self) -> str:
        return """
        Eu sou um agente de análise de dados e posso ajudá-lo a extrair insights valiosos do seu arquivo CSV. Aqui estão alguns exemplos do que você pode me pedir para fazer:
        **1. Análise Descritiva:**
        * _"Faça um resumo estatístico das colunas numéricas."_
        **2. Contagem e Frequência:**
        * _"Quantas categorias únicas existem na coluna 'produto' e qual a frequência de cada uma?"_
        **3. Filtragem e Ordenação:**
        * _"Liste as 5 vendas com o maior valor na coluna 'total'."_
        **4. Geração de Gráficos:**
        * _"Crie um gráfico de barras com a soma das vendas por região."_
        * _"Plote um histograma para ver a distribuição de idades."_
        * _"Crie um boxplot para a coluna 'preço' para identificar outliers."_
        **5. Correlação:**
        * _"Gere um heatmap de correlação para visualizar a relação entre as variáveis."_
        """

    def _python_executor(self, code: str) -> tuple[str, str]:
        if not isinstance(code, str) or not code.strip(): return "ERROR", "ERRO_INTERNO"
        output_buffer = io.StringIO(); sys.stdout = sys.stderr = output_buffer; plot_path = "temp_plot.png"
        if os.path.exists(plot_path): os.remove(plot_path)
        try:
            plt.close('all'); exec_globals = {"df": self.df, "pd": pd, "plt": plt, "sns": sns}; exec(code, exec_globals)
            output = output_buffer.getvalue()
            if os.path.exists(plot_path): return "IMAGE", plot_path
            return "TEXT", output if output.strip() else "Executado com sucesso."
        except Exception: return "ERROR", f"ERRO_EXECUCAO:\n{traceback.format_exc()}"
        finally: sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

    def run(self, user_query: str, chat_history: list) -> tuple[str, any]:
        try:
            request_options = {"timeout": 60}
            router_response = self.llm.generate_content(self._get_router_prompt(user_query), generation_config={"temperature": 0.0}, request_options=request_options)
            query_type = router_response.text.strip()

            if query_type == "agent_capabilities_query": return "TEXT", self._get_capability_examples()
            elif query_type == "general_knowledge": return "OUT_OF_SCOPE", "Desculpe, sou um especialista em análise de dados e só posso responder a perguntas sobre o CSV que você carregou."
            elif query_type == "dataframe_query":
                error_context = None
                for attempt in range(self.max_retries):
                    codegen_prompt = self._get_codegen_prompt(user_query, chat_history, error_context)
                    code_response = self.llm.generate_content(codegen_prompt, generation_config={"temperature": 0.1}, request_options=request_options)
                    generated_code = code_response.text.strip().removeprefix("```python").removesuffix("```").strip()
                    result_type, result_content = self._python_executor(generated_code)
                    if result_type != "ERROR":
                        if result_type == "IMAGE": return "IMAGE", (result_content, f"Aqui está o gráfico para '{user_query}'.")
                        else:
                            MAX_OUTPUT_LENGTH = 10000
                            if len(result_content) > MAX_OUTPUT_LENGTH: return "TEXT", (f"⚠️ **Atenção:** O resultado é muito longo...")
                            if '\n' in result_content.strip(): return "TEXT", f"```text\n{result_content}\n```"
                            else:
                                summary_response = self.llm.generate_content(self._get_summarizer_prompt(user_query, result_content), generation_config={"temperature": 0.7}, request_options=request_options)
                                return "TEXT", summary_response.text
                    else:
                        st.warning(f"Código gerado falhou... (Tentativa {attempt + 1}/{self.max_retries})"); error_context = {"code": generated_code, "error": result_content}
                return "ERROR", "O agente não conseguiu gerar um código funcional."
            else: return "ERROR", f"ERRO_INTERNO: Classificação inesperada: {query_type}"
        except core_exceptions.ResourceExhausted as e:
            error_text = str(e); wait_time = "cerca de 1 minuto"; retry_match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', error_text)
            if retry_match: wait_time = f"{int(retry_match.group(1)) + 1} segundos"
            return "ERROR", (f"**Atingimos o limite de requisições da API.**\n\nAguarde {wait_time} e tente novamente.")
        except Exception as e:
            if "Timeout" in str(e): return "ERROR", "A requisição à API demorou muito para responder (Timeout). Por favor, tente novamente."
            return "ERROR", f"Ocorreu um erro crítico no agente: {e}"

def main():
    st.set_page_config(page_title="Agente EDA Inteligente", layout="wide"); load_css()
    st.markdown('<div class="app-title">Agente Autônomo de Análise de Dados</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Faça upload de um CSV e converse com o agente.</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("⚙️ Configuração"); st.info("Este agente utiliza a API do Google Gemini.")
        st.markdown("---"); st.header("🚀 Como Usar"); st.caption("1. Faça o upload de um arquivo CSV.")
        st.caption("2. Use as ferramentas de Análise Rápida."); st.caption("3. Converse com o agente.")

    uploaded_file = st.file_uploader("Selecione um arquivo CSV para análise", type=["csv"], help="Tamanho máximo: 500MB.")
    if uploaded_file is not None:
        if st.session_state.get("last_file_id") != getattr(uploaded_file, 'file_id', id(uploaded_file)):
            trigger_rerun = False
            with st.spinner(f"Processando e limpando '{uploaded_file.name}'..."):
                try:
                    try: temp_df = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python')
                    except UnicodeDecodeError: uploaded_file.seek(0); temp_df = pd.read_csv(uploaded_file, encoding='latin-1', sep=None, engine='python')
                    for col in temp_df.select_dtypes(include=['object']).columns:
                        try:
                            cleaned_col = temp_df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                            temp_df[col] = pd.to_numeric(cleaned_col, errors='raise')
                        except (ValueError, TypeError, AttributeError): pass
                    st.session_state.df = temp_df
                    st.session_state.agent = None
                    st.session_state.messages = [{"role": "assistant", "content": "Olá! Seus dados foram carregados e limpos."}]
                    st.session_state.last_file_id = getattr(uploaded_file, 'file_id', id(uploaded_file))
                    trigger_rerun = True
                except Exception as e:
                    st.error(f"Erro ao ler o arquivo CSV: {e}"); st.session_state.df = None
            if trigger_rerun:
                st.rerun()

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"Dados prontos para análise ({df.shape[0]} linhas, {df.shape[1]} colunas).")
        with st.expander("🛠️ Ferramentas de Limpeza de Dados"):
            if st.button("Re-executar Limpeza Numérica"): pass
        st.subheader("Análise Rápida (AutoEDA)")
        with st.expander("🔍 Visualizar Amostra"): st.dataframe(df.head(10))
        with st.expander("ℹ️ Visão Geral do Dataset"):
            if st.button("Gerar Visão Geral", key="overview_btn"):
                st.subheader("Informações Gerais"); buffer = io.StringIO(); df.info(buf=buffer); st.text(buffer.getvalue())
                st.subheader("Resumo Estatístico"); st.dataframe(df.describe(include=np.number))
        st.markdown("---"); st.subheader("Converse com o Agente")
        if 'messages' not in st.session_state or not st.session_state.messages:
             st.session_state.messages = [{"role": "assistant", "content": "Olá! Seus dados estão prontos."}]
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"], unsafe_allow_html=True)
                if 'image_path' in message: st.image(message['image_path'])
        if user_prompt := st.chat_input("Faça sua pergunta..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user", avatar="👤"): st.markdown(user_prompt)
            with st.chat_message("assistant", avatar="🤖"):
                if "agent" not in st.session_state or st.session_state.agent is None:
                    try:
                        with st.spinner("Conectando ao agente de IA..."):
                            api_key = os.getenv("GEMINI_API_KEY"); genai.configure(api_key=api_key)
                            llm = genai.GenerativeModel('models/gemini-flash-latest')
                            st.session_state.agent = PandasAgent(df, llm)
                    except Exception as e:
                        error_msg = f"🚨 Falha ao inicializar o agente: {e}"; st.error(error_msg); st.stop()
                with st.spinner("🧠 O agente está pensando..."):
                    agent = st.session_state.agent
                    result_type, result_content = agent.run(user_prompt, st.session_state.messages)
                    response_message = {"role": "assistant", "content": result_content}
                    if result_type == "IMAGE":
                        img_path, summary = result_content
                        response_message.update({"content": summary, "image_path": img_path})
                    st.markdown(response_message["content"], unsafe_allow_html=True)
                    if "image_path" in response_message: st.image(response_message["image_path"])
                    st.session_state.messages.append(response_message)
    else:
        st.info("⬆️ Faça o upload de um arquivo CSV para começar.")

if __name__ == "__main__":

    main()
