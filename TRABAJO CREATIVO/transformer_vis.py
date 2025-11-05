
# Ejemplo de codigo para trabajar un texto en particular
# pip install seaborn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns

model = AutoModelForSequenceClassification.from_pretrained("tukx/fake-news-classificator", output_attentions=True)
text = "Trump kills martian people."

tokenizer = AutoTokenizer.from_pretrained("tukx/fake-news-classificator")
inputs = tokenizer(text, return_tensors="pt")



print('Inputs: ')
print(inputs)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

    attentions = outputs.attentions # Tupla de tensores de atención

#print('Atencion: ')
#print(attentions)
# Que relación tiene cada token con cada token?
# El primero 0 -> Primer batch
# Segundo 0 -> Primer head
# Tercer 0 -> P
attention_map = attentions[0][0][0]

print('Attention map?')
print(attention_map)

plt.figure(figsize=(10, 8))
sns.heatmap(attention_map.squeeze().detach().numpy(), cmap="viridis")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.title("Attention Map")
plt.show()

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
print('Logits: ')
print(logits)
print('Predicted Class Id: ')
print(predicted_class_id)
print('Predicted Class: ')
print(model.config.id2label[predicted_class_id])