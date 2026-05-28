from openai import OpenAI

# ==========================================
# CONFIGURACIÓN
# ==========================================

API_KEY = "sk-skl1uk31s3h9l4d37iixngj0ri3pw578qvssmavbkzmd6gc2"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.xiaomimimo.com/v1"
)

# ==========================================
# MENSAJE DEL SISTEMA
# ==========================================

messages = [
    {
        "role": "system",
        "content": (
            "Eres MiMo, un asistente inteligente desarrollado por Xiaomi. "
            "Siempre debes responder en español de forma clara y amigable."
        )
    }
]

print("===================================")
print("     CHAT XIAOMI MiMo IA")
print("===================================")
print("Escribe 'salir' para terminar.\n")

# ==========================================
# CHAT CONTINUO
# ==========================================

while True:

    pregunta = input("Tú: ")

    if pregunta.lower() == "salir":
        print("\nMiMo: ¡Hasta luego!")
        break

    # Agregar pregunta del usuario
    messages.append({
        "role": "user",
        "content": pregunta
    })

    try:

        completion = client.chat.completions.create(
            model="mimo-v2.5-pro",
            messages=messages,
            max_completion_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            stream=False
        )

        respuesta = completion.choices[0].message.content

        print(f"\nMiMo: {respuesta}\n")

        # Guardar respuesta en historial
        messages.append({
            "role": "assistant",
            "content": respuesta
        })

    except Exception as e:

        print("\nOCURRIÓ UN ERROR:")
        print(e)
        break