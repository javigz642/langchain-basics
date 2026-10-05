"""Ejercicio 2 · Reto: chat con memoria por session_id + herramientas sobre un CSV propio."""
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Ruta al CSV relativa a este archivo: funciona se ejecute desde donde se ejecute
RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos.csv"

SYSTEM_PROMPT = (
    "Eres un asistente de gestión de proyectos. Responde en español, de forma breve y profesional. "
    "Tienes acceso al registro de horas del equipo entre el 21/09/2026 y el 02/10/2026. "
    "El equipo está formado por Javier, Lucía, Marcos, Ana y Diego. "
    "Dispones de dos herramientas: 'total_horas', que devuelve las horas totales registradas por una persona, "
    "y 'hora_actual', que devuelve la hora actual. "
    "Usa las herramientas cuando las necesites, escribe los nombres exactamente como aparecen aquí "
    "y confía en sus resultados."
)

@tool
def total_horas(nombre: str) -> str:
    """Devuelve el número total de horas que ha registrado una persona del equipo en todos los proyectos."""
    df = pd.read_csv(RUTA_DATOS)
    horas_por_empleado = df.groupby("empleado")["horas"].sum()
    if nombre not in horas_por_empleado:
        disponibles = ", ".join(horas_por_empleado.index)
        return f"No hay registros de horas para '{nombre}'. Personas con registros: {disponibles}."
    return str(horas_por_empleado[nombre])


@tool
def hora_actual() -> str:
    """Devuelve la hora actual del sistema en formato HH:MM."""
    return datetime.now().strftime("%H:%M")


tools = [total_horas, hora_actual]
tools_por_nombre = {t.name: t for t in tools}
modelo = ChatOllama(model="llama3.1", temperature=0.2).bind_tools(tools)

# El "almacén" de memoria: una lista de mensajes por session_id (en producción: Redis o Postgres)
historiales: dict[str, list] = {}


def obtener_historial(session_id: str) -> list:
    # Cada sesión nueva recibe su PROPIA lista, que empieza con el system prompt
    if session_id not in historiales:
        historiales[session_id] = [SystemMessage(SYSTEM_PROMPT)]
    return historiales[session_id]

def turno(session_id: str, pregunta: str) -> str:
    # La lista devuelta es la misma que está en `historiales`: cada append queda guardado en la sesión
    mensajes = obtener_historial(session_id)
    mensajes.append(HumanMessage(pregunta))
    ai = modelo.invoke(mensajes)
    mensajes.append(ai)

    # while y no if: el modelo puede pedir varias rondas de tools antes de responder
    while ai.tool_calls:
        for llamada in ai.tool_calls:
            print(f"   (🛠️) Tool: {llamada['name']}({llamada['args']})")
            resultado = tools_por_nombre[llamada["name"]].invoke(llamada["args"])
            print(f"        -> {resultado}")
            # La tool call (en `ai`) y el ToolMessage quedan en el historial de la sesión
            mensajes.append(ToolMessage(content=str(resultado), tool_call_id=llamada["id"]))
        ai = modelo.invoke(mensajes)
        mensajes.append(ai)

    return ai.content



if __name__ == "__main__":
    conversaciones = {
        "javier": [
            "Buenos días, soy Javier, jefe de proyecto. ¿Cuántas horas he registrado en total?",
            "¿Y Lucía y Marcos? Quiero comparar la carga de trabajo del equipo.",
            "¿Qué hora es? Voy a anotarlo en el informe semanal.",
            "Recuérdame quién soy y cuál de los tres ha dedicado más horas.",
        ],
        #Otra sesión
        "lucia": [
            "Hola, soy Lucía. ¿Cuántas horas llevo registradas?",
            "¿Qué otras preguntas te he hecho en esta conversación?",
        ],
    }

    print("---INICIO---\n")
    for session_id, preguntas in conversaciones.items():
        print(f"===== Sesión '{session_id}' =====\n")
        for n, pregunta in enumerate(preguntas, 1):
            print(f"{n}º PREGUNTA: {pregunta}")
            respuesta = turno(session_id, pregunta)
            print(f"{n}º RESPUESTA: {respuesta}\n")

    for session_id, mensajes in historiales.items():
        print(f"Mensajes guardados en la sesión '{session_id}': {len(mensajes)}")
    print("---FIN---")
