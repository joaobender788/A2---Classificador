import pandas as pd
from pickle import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# TRAVA DE SEGURANÇA OBRIGATÓRIA PARA WINDOWS QUANDO SE USA n_jobs=-1
if __name__ == '__main__':
    print("=== (BANCO AGORAVAI) ===")

    # 1. Carregar a base de dados
    try:
        dados = pd.read_csv('default_of_credit_card_clients.csv', sep=';')
        print("Base de dados carregada com sucesso!")
    except FileNotFoundError:
        print("Erro: O arquivo 'default_of_credit_card_clients.csv' não foi encontrado.")
        exit()

    # Otimização de volume (Amostra de 30% para performance) - MANTIDO
    dados = dados.sample(frac=0.30, random_state=42).reset_index(drop=True)

    # Padronizar nomes de colunas
    dados.columns = dados.columns.str.upper().str.strip()

    if 'ID' in dados.columns:
        dados = dados.drop(columns=['ID'])

    # 2. Conversão Estrita de Tipos (Garantindo que números sejam números)
    colunas_numericas = [
        'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
        'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    for col in colunas_numericas:
        if col in dados.columns:
            dados[col] = pd.to_numeric(dados[col], errors='coerce')
    dados[colunas_numericas] = dados[colunas_numericas].fillna(0)

    # 3. Engenharia de Atributos (MODIFICADO: Preservando a gravidade do atraso)
    # Se o valor for > 0, mantém o número de meses atrasados. Se for <= 0 (em dia/adiantado), vira 0.
    colunas_pay = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for col in colunas_pay:
        if col in dados.columns:
            dados[col] = pd.to_numeric(dados[col], errors='coerce').fillna(0)
            dados[col] = dados[col].apply(lambda x: x if x > 0 else 0)

    # Limpeza e Encoding das colunas de texto
    colunas_texto = ['SEX', 'EDUCATION', 'MARRIAGE']
    for col in colunas_texto:
        if col in dados.columns:
            dados[col] = dados[col].astype(str).str.strip()

    dados_processados = pd.get_dummies(dados, columns=colunas_texto, drop_first=True)

    # Converter colunas booleanas para inteiros (0 ou 1)
    colunas_bool = dados_processados.select_dtypes(include=['bool']).columns
    dados_processados[colunas_bool] = dados_processados[colunas_bool].astype(int)

    # Separar atributos e classe alvo
    nome_classe = 'DEFAULT PAYMENT NEXT MONTH'
    dados_atributos = dados_processados.drop(columns=[nome_classe])
    dados_classe = dados_processados[nome_classe]

    # Salvar a lista exata de colunas do treino para a inferência usar
    colunas_treino = list(dados_atributos.columns)

    # 4. Divisão de Treino e Teste (Sem SMOTE!)
    atributos_train, atributos_teste, classe_train, classe_test = train_test_split(
        dados_atributos, dados_classe, test_size=0.3, random_state=42, stratify=dados_classe
    )

    # 5. Normalização dos Dados REMOVIDA
    # Modelos de Árvore (Random Forest) não sofrem impacto com escalas diferentes.

    # 6. Treinamento do Modelo (MODIFICADO: Hiperparametrização reintegrada)
    print("\nOtimizando hiperparâmetros do Random Forest com pesos balanceados...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }

    # O class_weight='balanced' foi mantido, é uma ótima escolha!
    modelo_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    otimizacao = RandomizedSearchCV(
        estimator=modelo_base,
        param_distributions=rf_params,
        n_iter=10,
        cv=3,
        scoring='f1_macro', # Focando na métrica correta para desbalanceamento
        random_state=42,
        n_jobs=-1
    )
    otimizacao.fit(atributos_train, classe_train)

    modelo_campeao = otimizacao.best_estimator_
    print(f"Melhores parâmetros: {otimizacao.best_params_}")

    # 7. Avaliação Robusta 
    pred = modelo_campeao.predict(atributos_teste)
    acuracia = accuracy_score(classe_test, pred)
    f1 = f1_score(classe_test, pred, average='macro')

    print(f"\nAcurácia do modelo no Teste: {acuracia:.2%}")
    print(f"F1-Score (Macro) no Teste: {f1:.4f}")
    print("\nRelatório de Classificação:\n", classification_report(classe_test, pred))

    # 8. Salvando os Artefatos 
    dump(modelo_campeao, open('melhor_modelo_credito.pkl', 'wb'))
    dump(colunas_treino, open('colunas_treino.pkl', 'wb'))
    print("Novos artefatos limpos salvos com sucesso (Modelo e Colunas)!\n")
    print("Auditoria finalizada. Se os scores condizem com os perfis, sua inferência está validada!")
    print("=" * 65)