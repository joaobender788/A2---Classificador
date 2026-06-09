import pandas as pd
from pickle import load

print("=== INFERÊNCIA (BANCO AGORAVAI) ===")

# 1. Carregar novos artefatos salvos pelo treino (Sem o Scaler)
try:
    modelo = load(open('melhor_modelo_credito.pkl', 'rb'))
    colunas_treino = load(open('colunas_treino.pkl', 'rb'))
    print("Artefatos de produção carregados com sucesso.\n")
except FileNotFoundError:
    print("Erro: Execute o script de treinamento primeiro para gerar os arquivos .pkl")
    exit()

# 2. Criação do Banco de Dados de Teste Calibrado
cenarios_clientes = [
    {
        'ID_TESTE': 'Cliente 01 - Excelente Pagador',
        'LIMIT_BAL': 500000, 'SEX': 'F', 'EDUCATION': 'Graduate School', 'MARRIAGE': 'Married', 'AGE': 35,
        'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0, # Em dia
        'BILL_AMT1': 1000, 'BILL_AMT2': 1000, 'BILL_AMT3': 1000, 'BILL_AMT4': 1000, 'BILL_AMT5': 1000, 'BILL_AMT6': 1000,
        'PAY_AMT1': 1000, 'PAY_AMT2': 1000, 'PAY_AMT3': 1000, 'PAY_AMT4': 1000, 'PAY_AMT5': 1000, 'PAY_AMT6': 1000
    },
    {
        'ID_TESTE': 'Cliente 02 - Alto Risco Crônico',
        'LIMIT_BAL': 10000, 'SEX': 'M', 'EDUCATION': 'Middle School', 'MARRIAGE': 'Single', 'AGE': 22,
        'PAY_0': 2, 'PAY_2': 3, 'PAY_3': 2, 'PAY_4': 2, 'PAY_5': 2, 'PAY_6': 2, # Atrasos de 2 e 3 meses
        'BILL_AMT1': 9500, 'BILL_AMT2': 9500, 'BILL_AMT3': 9500, 'BILL_AMT4': 9500, 'BILL_AMT5': 9500, 'BILL_AMT6': 9500,
        'PAY_AMT1': 0, 'PAY_AMT2': 0, 'PAY_AMT3': 0, 'PAY_AMT4': 0, 'PAY_AMT5': 0, 'PAY_AMT6': 0
    },
    {
        'ID_TESTE': 'Cliente 03 - Risco Moderado / Oscilante',
        'LIMIT_BAL': 80000, 'SEX': 'M', 'EDUCATION': 'University', 'MARRIAGE': 'Married', 'AGE': 40,
        'PAY_0': 1, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0, # Um atraso leve de 1 mês
        'BILL_AMT1': 20000, 'BILL_AMT2': 15000, 'BILL_AMT3': 10000, 'BILL_AMT4': 5000, 'BILL_AMT5': 4000, 'BILL_AMT6': 2000,
        'PAY_AMT1': 1000, 'PAY_AMT2': 5000, 'PAY_AMT3': 5000, 'PAY_AMT4': 2000, 'PAY_AMT5': 2000, 'PAY_AMT6': 2000
    }
]

identificadores = [cliente['ID_TESTE'] for cliente in cenarios_clientes]

# Converter chaves para maiúsculo para bater com o treino
for cenario in cenarios_clientes:
    chaves_antigas = list(cenario.keys())
    for k in chaves_antigas:
        if k != 'ID_TESTE':
            cenario[k.upper()] = cenario.pop(k)

# 3. Processar cada cliente individualmente
clientes_processados = []
for cenario in cenarios_clientes:
    dados_cliente = cenario.copy()
    dados_cliente.pop('ID_TESTE')
    
    df_unico = pd.DataFrame([dados_cliente])
    df_proc = pd.get_dummies(df_unico, columns=['SEX', 'EDUCATION', 'MARRIAGE'])
    
    # Alinhamento estrito de colunas
    df_alinhado = df_proc.reindex(columns=colunas_treino, fill_value=0)
    df_alinhado = df_alinhado.astype(float)
    
    clientes_processados.append(df_alinhado)

df_final = pd.concat(clientes_processados, ignore_index=True)

# Predizer classes e obter probabilidades reais (Direto no modelo, sem scaler)
probabilidades = modelo.predict_proba(df_final)

# 4. Relatório de Auditoria Final (MODIFICADO: Regra de Negócio de Risco)
print("=" * 65)
print("     BANCO AGORAVAI - RELATÓRIO DE AUDITORIA DA INFERÊNCIA     ")
print("=" * 65)

for idx, nome_cliente in enumerate(identificadores):
    prob_inadimplencia = probabilidades[idx][1] * 100
    prob_adimplencia = probabilidades[idx][0] * 100
    
    # Regra de negócio (Score)
    if prob_inadimplencia >= 70:
        nivel_risco = "Alto"
        decisao = "NEGAR CRÉDITO"
    elif prob_inadimplencia >= 30:
        nivel_risco = "Moderado"
        decisao = "APROVAR COM RESSALVAS"
    else:
        nivel_risco = "Baixo"
        decisao = "APROVAR CRÉDITO"
    
    print(f"{nome_cliente}")
    print(f"   -> Decisão do Sistema: {decisao} ({nivel_risco})")
    print(f"   -> Score de Risco (Prob. Inadimplência): {prob_inadimplencia:.2f}%")
    print(f"   -> Confiança de Pagamento: {prob_adimplencia:.2f}%")
    print("-" * 65)

print("Auditoria finalizada. Pipeline refeito e robusto!")
print("=" * 65)