## COMP0427 - INTELIGÊNCIA ARTIFICIAL (2024.2 - T01)
## Professor: HENDRIK TEIXEIRA MACEDO

## TRAB. DE INTELIGÊNCIA ARTIFICIAL – A1

## Grupo:

Déborah Abreu Sales
deborahsales@academico.ufs.br

Leticia da Mata Cavalcanti
leticiacavalcant@academico.ufs.br

Harrison Santos Siqueira
harriison@academico.ufs.br

Felipe Rodrigues Santana
felipe.santana@dcomp.ufs.br

Nicolas Vieira dos Santos
nicolas.vieira@dcomp.ufs.br

# **Instruções para Executar o Notebook**

## **Pré-requisitos**
Antes de executar o notebook, certifique-se de ter o ambiente configurado corretamente:

### **1. Instalar Dependências**
Execute os seguintes comandos para instalar as bibliotecas necessárias:

```bash
!apt-get install -y xvfb x11-utils
!pip install gym[atari] stable-baselines3 opencv-python gym[accept-rom-license] pyvirtualdisplay
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

### **2. Configurar o Ambiente**
Se estiver rodando em um ambiente como Google Colab, execute:

```python
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
```
Isso evita problemas com a renderização gráfica do ambiente de jogo.

## **Como Rodar o Treinamento**

### **1. Baixar e Abrir o Notebook**
   - Faça o upload do notebook no Google Colab ou execute localmente com Jupyter Notebook.
  
### **2. Executar as Células em Ordem**
   - Rode todas as células na sequência para importar as bibliotecas, configurar o ambiente, definir a rede neural DQN e iniciar o treinamento.

### **3. Treinar o Modelo**
   - O treinamento pode levar algum tempo, dependendo da capacidade da sua GPU.
   - Parâmetros ajustáveis incluem `num_episodes`, `learning_rate`, `epsilon_decay` e `batch_size`.

### **4. Salvar e Carregar o Modelo**
   - Se houver modelos salvos, o script tentará carregá-los automaticamente:
   
   ```python
   model_dir = "/content/drive/MyDrive/Salvo do Chrome/DqnModel"
   ```
   
   - Para salvar um modelo treinado, use:
   
   ```python
   torch.save(model.state_dict(), "dqn_model.pth")
   ```

### **5. Testar o Modelo**
   - Após o treinamento, o modelo pode ser testado jogando um episódio e visualizando as ações.

---
Caso tenha dúvidas ou precise modificar hiperparâmetros, edite as variáveis no código.



