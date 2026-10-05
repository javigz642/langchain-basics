from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

#1. PROMPT: La instrucción se escribe una vez, con {texto} es el hueco que rellenamos al invocar
prompt = ChatPromptTemplate.from_messages([
    ("system", "Resume el texto en EXACTAMENTE 3 viñetas. El formato de salida es por cada viñeta empezar con '-'. Sin introducción ni cierre."),
    ("human", "{texto}")
])

#2. Modelo: el ChatModel local. Cambiar de proveedor = cambiar solo esta línea
modelo = ChatOllama(model="llama3.1", temperature=0)

#3. PARSER: el modelo devuelve un AIMessage; el parser se queda con el texto limpio
parser = StrOutputParser()

# LCEL
cadena = prompt | modelo | parser

if __name__ == "__main__":
    textos = {
        "Texto 1 · vehículos eléctricos": (
            "Los vehículos eléctricos están ganando presencia en las ciudades gracias a la mejora de las baterías y al "
            "aumento de los puntos de recarga. Muchos conductores valoran su menor coste de mantenimiento y la ausencia "
            "de emisiones directas durante la conducción. Sin embargo, todavía existen retos: el precio de algunos "
            "modelos, el tiempo necesario para recargar y la disponibilidad de cargadores en determinadas zonas. Por eso "
            "fabricantes y administraciones están invirtiendo en nuevas infraestructuras para facilitar su expansión."),

        "Texto 2 · agricultura urbana": (
            "La agricultura urbana se está extendiendo en muchas ciudades como una forma de producir alimentos cerca de "
            "los consumidores. Huertos comunitarios, terrazas y pequeños invernaderos permiten cultivar verduras y "
            "hierbas aprovechando espacios que antes tenían poco uso. Su principal dificultad es disponer de suficiente "
            "superficie, agua y tiempo para mantener los cultivos, especialmente en zonas muy densas. Aun así, muchos "
            "proyectos la consideran una herramienta útil para fomentar la sostenibilidad y la participación vecinal."),

        "Texto 3 · inteligencia artificial en la educación": (
            "La inteligencia artificial está empezando a transformar la educación mediante herramientas capaces de "
            "adaptar ejercicios, explicar conceptos y ofrecer apoyo personalizado a los estudiantes. Los docentes pueden "
            "utilizarla para preparar materiales y detectar dificultades de aprendizaje con mayor rapidez. Sin embargo, "
            "también plantea desafíos relacionados con la privacidad, la dependencia tecnológica y el uso responsable "
            "de estas herramientas. Por ello, muchos centros educativos buscan integrarla como apoyo sin sustituir el "
            "papel fundamental del profesorado.")
    }

    for n, texto in enumerate(textos):
        print(f"\n{texto}")
        salida = cadena.invoke(textos[texto])
        print(salida)
  
    
