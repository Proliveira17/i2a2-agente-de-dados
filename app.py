

REGRAS IMPORTANTES:
1.  Gere APENAS o código Python necessário para responder à pergunta. Não inclua texto explicativo, comentários desnecessários ou a palavra "python" no início.
2.  Use `print()` para exibir saídas de texto, como tabelas, números ou estatísticas. O resultado do `print()` será capturado.
3.  Para gerar gráficos (barras, linhas, histogramas, etc.), use `matplotlib.pyplot` (aliased como `plt`) ou `seaborn` (aliased como `sns`). O gráfico será salvo e exibido automaticamente. NÃO use `plt.show()`.
4.  O DataFrame está sempre disponível na variável `df`.
5.  Se a pergunta for complexa, quebre o problema em etapas no seu código.

Gere o código Python agora.


    def _get_summarizer_prompt(self, user_query: str, result: str) -> str:
        return (f"Você é um assistente de análise de dados. A pergunta do usuário foi: '{user_query}'.\nO código executado produziu o seguinte resultado: '{result}'.\n\nExplique este resultado para o usuário de forma clara, objetiva e em português.")

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
        if not isinstance(code, str) or not code.strip():
            return "ERROR", "ERRO_INTERNO: Código gerado estava vazio."

        output_buffer = io.StringIO()
        sys.stdout = sys.stderr = output_buffer
        plot_path = "temp_plot.png"
        if os.path.exists(plot_path):
            os.remove(plot_path)

        try:
            plt.close('all')
            exec_globals = {"df": self.df, "pd": pd, "plt": plt, "sns": sns, "np": np}
            exec(code, exec_globals)
            output = output_buffer.getvalue()

            if plt.get_fignums():
                plt.savefig(plot_path, format='png', bbox_inches='tight')
                return "IMAGE", plot_path
            else:
                return "TEXT", output if output.strip() else "Código executado com sucesso, mas não produziu texto."
        except Exception:
            return "ERROR", f"ERRO_EXECUCAO:\n{traceback.format_exc()}"
        finally:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

    def run(self, user_query: str, chat_history: list) -> tuple[str, any]:
        try:
            request_options = {"timeout": 90}
            router_response = self.llm.generate_content(self._get_router_prompt(user_query), generation_config={"temperature": 0.0}, request_options=request_options)
            query_type = router_response.text.strip()

            if query_type == "agent_capabilities_query":
                return "TEXT", {"summary": self._get_capability_examples(), "code": None}
            elif query_type == "general_knowledge":
                return "OUT_OF_SCOPE", {"summary": "Desculpe, sou um especialista em análise de dados e só posso responder a perguntas sobre o CSV que você carregou.", "code": None}
            elif query_type == "dataframe_query":
                error_context = None
                for attempt in range(self.max_retries):
                    codegen_prompt = self._get_codegen_prompt(user_query, chat_history, error_context)
                    code_response = self.llm.generate_content(codegen_prompt, generation_config={"temperature": 0.1}, request_options=request_options)
                    generated_code = code_response.text.strip().removeprefix("```python").removesuffix("```").strip()

                    result_type, result_content = self._python_executor(generated_code)

                    if result_type != "ERROR":
                        if result_type == "IMAGE":
                            return "IMAGE", {"summary": f"Aqui está o gráfico que preparei para responder a: '{user_query}'.", "code": generated_code, "path": result_content}
                        else:
                            # MELHORIA 2: LÓGICA DE SUMARIZAÇÃO E TRUNCAMENTO
                            # Trata saídas muito longas para não poluir a tela e mantém a sumarização para saídas curtas.
                            MAX_OUTPUT_LENGTH = 5000
                            if len(result_content) > MAX_OUTPUT_LENGTH:
                                truncated_result = result_content[:MAX_OUTPUT_LENGTH] + "\n\n... (resultado truncado por ser muito longo) ..."
                                summary = f"O código executado com sucesso e produziu um resultado muito longo. Aqui estão os primeiros caracteres:\n\n```text\n{truncated_result}\n```"
                                return "TEXT", {"summary": summary, "code": generated_code}

                            # Se for uma saída com cara de tabela/dado bruto, mostre-a diretamente.
                            if '\n' in result_content.strip() or len(result_content) > 100:
                                return "TEXT", {"summary": f"```text\n{result_content}\n```", "code": generated_code}
                            else: # Caso contrário, é um resultado curto que vale a pena explicar.
                                summary_response = self.llm.generate_content(self._get_summarizer_prompt(user_query, result_content), generation_config={"temperature": 0.7}, request_options=request_options)
                                return "TEXT", {"summary": summary_response.text, "code": generated_code}
                    else:
                        st.warning(f"O código gerado falhou. Analisando o erro e tentando novamente... (Tentativa {attempt + 1}/{self.max_retries})")
                        time.sleep(1)
                        error_context = {"code": generated_code, "error": result_content}

                final_error_msg = f"O agente não conseguiu gerar um código funcional após {self.max_retries} tentativas. \n\n**Último erro:**\n ```\n{error_context['error']}\n```"
                return "ERROR", {"summary": final_error_msg, "code": error_context['code']}
            else:
                return "ERROR", {"summary": f"ERRO_INTERNO: Classificação inesperada da pergunta: {query_type}", "code": None}

        except core_exceptions.ResourceExhausted as e:
            error_text = str(e)
            wait_time = "cerca de 1 minuto"
            retry_match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', error_text)
            if retry_match:
                wait_time = f"{int(retry_match.group(1)) + 1} segundos"
            return "ERROR", {"summary": f"**Atingimos o limite de requisições da API do Gemini.**\n\nAguarde {wait_time} e tente novamente.", "code": None}
        except Exception as e:
            if "Timeout" in str(e):
                return "ERROR", {"summary": "A requisição à API demorou muito para responder (Timeout). Por favor, tente novamente.", "code": None}
            return "ERROR", {"summary": f"Ocorreu um erro crítico no agente: {e}", "code": None}

# DENTRO DA CÉLULA 2 (app.py), SUBSTITUA A FUNÇÃO main() POR ESTA VERSÃO COM O BOTÃO:
def main():
    st.set_page_config(page_title="Agente EDA Inteligente", layout="wide")
    load_css()
    st.markdown('<div class="app-title">Agente Autônomo de Análise de Dados</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Faça upload de um CSV e converse com o agente para extrair insights.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configurações")
        st.info("Este agente utiliza a API do Google Gemini para gerar e analisar dados.")

        selected_model = st.selectbox(
            "Escolha o modelo de IA:",
            ("models/gemini-flash-latest", "gemini-pro"), 
            index=0,
            format_func=lambda x: "Gemini Flash (Rápido)" if "flash" in x else "Gemini Pro (Padrão)",
            help="Modelos padrão com alta compatibilidade. Flash é mais rápido, Pro é mais robusto para perguntas complexas."
        )
        st.session_state.selected_model = selected_model
        
        st.warning("**Aviso:** Este agente gera e executa código Python. Use com dados de fontes confiáveis.")

        st.markdown("---")

        # MELHORIA DO USUÁRIO: Botão para Limpar o Chat
        if st.button("🧹 Limpar Chat"):
            # Reinicia a lista de mensagens para o estado inicial
            st.session_state.messages = [{"role": "assistant", "content": "Olá! Seus dados foram carregados. O que você gostaria de analisar?", "code": None}]
            # Força o recarregamento da página para exibir a limpeza
            st.rerun()

        st.markdown("---")
        st.header("🚀 Como Usar")
        st.caption("1. Faça o upload de um arquivo CSV.")
        st.caption("2. Use as ferramentas de Análise Rápida.")
        st.caption("3. Converse com o agente no chat.")

    uploaded_file = st.file_uploader("Selecione um arquivo CSV para análise", type=["csv"], help="Tamanho máximo: 500MB.")

    if uploaded_file is not None:
        file_identifier = getattr(uploaded_file, 'file_id', id(uploaded_file))
        
        if st.session_state.get("last_file_id") != file_identifier:
            trigger_rerun = False
            with st.spinner(f"Processando e limpando '{uploaded_file.name}'..."):
                try:
                    try:
                        temp_df = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        temp_df = pd.read_csv(uploaded_file, encoding='latin-1', sep=None, engine='python')

                    for col in temp_df.select_dtypes(include=['object']).columns:
                        try:
                            cleaned_col = temp_df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                            temp_df[col] = pd.to_numeric(cleaned_col, errors='raise')
                        except (ValueError, TypeError, AttributeError):
                            pass

                    st.session_state.df = temp_df
                    st.session_state.agent = None
                    st.session_state.messages = [{"role": "assistant", "content": "Olá! Seus dados foram carregados. O que você gostaria de analisar?", "code": None}]
                    st.session_state.last_file_id = file_identifier
                    trigger_rerun = True

                except Exception as e:
                    st.error(f"Erro ao ler o arquivo CSV: {e}")
                    st.session_state.df = None
            
            if trigger_rerun:
                st.rerun()

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"Dados prontos para análise ({df.shape[0]} linhas, {df.shape[1]} colunas).")
        
        with st.expander("🔍 Visualizar Amostra dos Dados"):
            st.dataframe(df.head(10))

        st.markdown("---")
        st.subheader("💬 Converse com o Agente")

        # Verifica se há um dataframe carregado antes de iniciar o chat
        if 'messages' not in st.session_state and df is not None:
             st.session_state.messages = [{"role": "assistant", "content": "Olá! Seus dados foram carregados. O que você gostaria de analisar?", "code": None}]
        elif 'messages' not in st.session_state:
             st.session_state.messages = []


        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"], unsafe_allow_html=True)
                if 'image_path' in message:
                    st.image(message['image_path'])
                if 'code' in message and message['code']:
                    with st.expander("Ver código gerado"):
                        st.code(message['code'], language='python')

        if user_prompt := st.chat_input("Ex: 'Qual a média de idade?' ou 'Crie um gráfico de vendas por mês'"):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🧠 O agente está pensando..."):
                    if "agent" not in st.session_state or st.session_state.agent is None:
                        try:
                            api_key = os.getenv("GEMINI_API_KEY")
                            if not api_key:
                                st.error("Chave de API GEMINI_API_KEY não encontrada.")
                                st.stop()
                            genai.configure(api_key=api_key)
                            llm = genai.GenerativeModel(st.session_state.selected_model)
                            st.session_state.agent = PandasAgent(df, llm)
                        except Exception as e:
                            st.error(f"🚨 Falha ao inicializar o agente de IA: {e}")
                            st.stop()
                    
                    agent = st.session_state.agent
                    result_type, result_data = agent.run(user_prompt, st.session_state.messages)

                    response_message = {"role": "assistant"}
                    if result_type == "IMAGE":
                        response_message.update({
                            "content": result_data["summary"],
                            "code": result_data["code"],
                            "image_path": result_data["path"]
                        })
                    else:
                         response_message.update({
                            "content": result_data["summary"],
                            "code": result_data["code"]
                        })

                    st.markdown(response_message["content"], unsafe_allow_html=True)
                    if "image_path" in response_message:
                        st.image(response_message["image_path"])
                    if "code" in response_message and response_message["code"]:
                         with st.expander("Ver código gerado"):
                            st.code(response_message['code'], language='python')
                    
                    st.session_state.messages.append(response_message)
    else:
        st.info("⬆️ Faça o upload de um arquivo CSV na barra lateral para começar a análise.")
if __name__ == "__main__":
    main()

