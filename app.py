from flask import Flask, render_template, request, Response
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.io as pio
import io
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar los datos ingresados
@app.route('/calcular', methods=['POST'])
def calcular():
    mensaje_recomendacion = None

    # Obtener los pesos de los criterios
    peso_precio = float(request.form['peso_precio'])
    peso_tiempo = float(request.form['peso_tiempo'])
    peso_calidad = float(request.form['peso_calidad'])

    # Validar que los pesos sumen 1
    suma_pesos = peso_precio + peso_tiempo + peso_calidad
    if abs(suma_pesos - 1.0) > 0.01:
        return "Error: La suma de los pesos debe ser 1.", 400

    # Obtener los datos de los proveedores
    proveedores = []
    i = 0
    while f'nombre_{i}' in request.form:
        nombre = request.form[f'nombre_{i}']
        precio = float(request.form[f'precio_{i}'])
        tiempo_entrega = float(request.form[f'tiempo_{i}'])
        calidad = float(request.form[f'calidad_{i}'])

        # Validar calidad, precio y tiempo de entrega
        if calidad < 1 or calidad > 10:
            return "Error: La calidad debe estar entre 1 y 10.", 400
        if precio < 0 or precio > 1000000:
            return "Error: El precio debe estar entre 0 y 1,000,000.", 400
        if tiempo_entrega < 1 or tiempo_entrega > 10:
            return "Error: El tiempo de entrega debe ser entre 1 y 10 días hábiles.", 400

        proveedores.append({"nombre": nombre, "precio": precio, "tiempo_entrega": tiempo_entrega, "calidad": calidad})
        i += 1

    # Convertir a DataFrame
    df = pd.DataFrame(proveedores)

    # Normalizar y calcular puntuaciones
    df['precio_normalizado'] = 1 - (df['precio'] - df['precio'].min()) / np.clip((df['precio'].max() - df['precio'].min()), 1e-10, None)
    df['tiempo_normalizado'] = 1 - (df['tiempo_entrega'] - df['tiempo_entrega'].min()) / np.clip((df['tiempo_entrega'].max() - df['tiempo_entrega'].min()), 1e-10, None)
    df['calidad_normalizada'] = (df['calidad'] - df['calidad'].min()) / np.clip((df['calidad'].max() - df['calidad'].min()), 1e-10, None)

    # Calcular puntuación ponderada
    df['puntuacion'] = (df['precio_normalizado'] * peso_precio +
                        df['tiempo_normalizado'] * peso_tiempo +
                        df['calidad_normalizada'] * peso_calidad)
    df['puntuacion'] = df['puntuacion'].round(2)

    # Identificar el mejor proveedor basado en la puntuación
    mejor_proveedor = df.loc[df['puntuacion'].idxmax()]

    # Árbol de decisión: Clasificación de calidad
    X = df[['precio', 'tiempo_entrega', 'calidad']]
    y = (df['calidad'] >= 8).astype(int)  # Calidad alta si >= 8
    
    # Entrenar el árbol de decisión
    arbol_decision = DecisionTreeClassifier()
    arbol_decision.fit(X, y)
    
    # Predicciones
    df['prediccion_calidad'] = arbol_decision.predict(X)
    df['calidad_clasificada'] = df['prediccion_calidad'].apply(lambda x: 'Alta Calidad' if x == 1 else 'Baja Calidad')

    # Generar gráfico del árbol de decisión
    plt.figure(figsize=(12, 8))
    tree.plot_tree(arbol_decision, feature_names=['Precio', 'Tiempo de Entrega', 'Calidad'], class_names=['Baja Calidad', 'Alta Calidad'], filled=True)
    plt.tight_layout()
    plt.savefig('static/arbol_decision.png')
    plt.close()

    # Gráfico de puntuaciones usando Plotly
    fig_puntuaciones = px.bar(df, x='nombre', y='puntuacion', title='Puntuaciones Ponderadas de Proveedores',
                            labels={'nombre': 'Proveedores', 'puntuacion': 'Puntuación Ponderada'})
    mejor_proveedor_nombre = mejor_proveedor["nombre"]
    fig_puntuaciones.add_hline(y=mejor_proveedor['puntuacion'], line_dash='dash',
                                annotation_text=f'Mejor Proveedor: {mejor_proveedor_nombre}',
                                annotation_position="top right")
    pio.write_image(fig_puntuaciones, 'static/plotly_images/grafico_proveedores.png')

    # Gráfico de Radar
    fig_radar = crear_grafico_radar(df)
    pio.write_image(fig_radar, 'static/plotly_images/grafico_radar.png')

    # Gráfico de Dispersión
    fig_dispersión = px.scatter(df, x='precio', y='calidad', color='nombre',
                                title='Relación entre Precio y Calidad', 
                                labels={'precio': 'Precio', 'calidad': 'Calidad'})
    pio.write_image(fig_dispersión, 'static/plotly_images/grafico_dispersión.png')

    # Gráfico de Boxplot
    fig_box = px.box(df, y='puntuacion', points='all', title='Distribución de Puntuaciones de Proveedores',
                    labels={'puntuacion': 'Puntuación'})
    pio.write_image(fig_box, 'static/plotly_images/grafico_boxplot.png')

    # Lógica para recomendar un proveedor
    recomendacion_final = df[(df['puntuacion'] == df['puntuacion'].max()) & (df['calidad_clasificada'] == 'Alta Calidad')]

    # Verificar si hay un proveedor que cumpla con los criterios
    if not recomendacion_final.empty:
        recomendacion_final = recomendacion_final.iloc[0]  # Tomar el primer proveedor que cumpla
        mensaje_recomendacion = f"Se eligió al proveedor {mejor_proveedor['nombre']} debido a su alta puntuación que fue de {mejor_proveedor['puntuacion']} y su calidad que es de {mejor_proveedor['calidad']}."
    else:
        # Si no hay proveedor que cumpla, establecer un mensaje
        recomendacion_final = None  # No se recomienda a nadie
        mensaje_recomendacion = "Para hacer la recomendación tanto el puntaje como la alta calidad son necesarios y ninguno de estos proveedores lo cumplen, por eso no recomendamos a nadie. Sigue buscando mejores proveedores."

    app.config['RESULTADOS_DF'] = df

    # Pasar el mensaje a la plantilla de resultados
    return render_template('resultados.html', df=df, mejor_proveedor=mejor_proveedor, 
                        recomendacion_final=recomendacion_final, 
                        mensaje_recomendacion=mensaje_recomendacion,  # Agregar el mensaje
                        imagen_puntuaciones='static/plotly_images/grafico_proveedores.png', 
                        imagen_arbol='static/arbol_decision.png',
                        imagen_radar='static/plotly_images/grafico_radar.png',
                        imagen_dispersión='static/plotly_images/grafico_dispersión.png',
                        imagen_box='static/plotly_images/grafico_boxplot.png')

def crear_grafico_radar(df):
    df_melted = df.melt(id_vars='nombre', value_vars=['precio', 'tiempo_entrega', 'calidad'], 
                        var_name='Criterio', value_name='Valor')
    
    fig = px.line_polar(df_melted, r='Valor', theta='Criterio', color='nombre',
                        line_close=True, title='Gráfico de Radar de Proveedores')
    fig.update_traces(fill='toself')
    return fig

@app.route('/descargar_pdf')
def descargar_pdf():
    df = app.config['RESULTADOS_DF']  # Obtener el DataFrame almacenado
    response = Response(content_type='application/pdf')
    response.headers['Content-Disposition'] = 'attachment; filename=resultados_proveedores.pdf'
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=landscape(letter))  # Cambiar a orientación horizontal
    width, height = landscape(letter)

    # Agregar título
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, height - 50, "Resultados de Evaluación de Proveedores")

    # Agregar el mejor proveedor
    p.setFont("Helvetica", 12)
    mejor_proveedor = df.loc[df['puntuacion'].idxmax()]
    p.drawString(100, height - 70, "Mejor Proveedor: " + mejor_proveedor['nombre'])

    # Agregar encabezados de tabla
    p.setFont("Helvetica-Bold", 10)
    p.drawString(50, height - 100, "Nombre")
    p.drawString(150, height - 100, "Precio")
    p.drawString(250, height - 100, "Tiempo de Entrega")
    p.drawString(350, height - 100, "Calidad")
    p.drawString(450, height - 100, "Puntuación")
    p.drawString(550, height - 100, "Clasificación")

    # Dibujar línea horizontal para encabezados
    p.line(40, height - 102, width - 40, height - 102)

    # Agregar filas de datos
    y = height - 110
    p.setFont("Helvetica", 8)  # Cambiar a un tamaño de fuente más pequeño
    for index, row in df.iterrows():
        if y < 40:  # Si el espacio se está acabando, añadir nueva página
            p.showPage()  # Cambiar a una nueva página
            p.setFont("Helvetica-Bold", 10)
            p.drawString(50, height - 50, "Nombre")
            p.drawString(150, height - 50, "Precio")
            p.drawString(250, height - 50, "Tiempo de Entrega")
            p.drawString(350, height - 50, "Calidad")
            p.drawString(450, height - 50, "Puntuación")
            p.drawString(550, height - 50, "Clasificación")
            p.line(40, height - 52, width - 40, height - 52)  # Línea para encabezados
            y = height - 70  # Reiniciar posición Y en la nueva página

        # Dibujar cada fila de la tabla
        p.drawString(50, y, row['nombre'][:20])  # Limitar el nombre a 20 caracteres
        p.drawString(150, y, str(row['precio']))
        p.drawString(250, y, str(row['tiempo_entrega']))
        p.drawString(350, y, str(row['calidad']))
        p.drawString(450, y, str(row['puntuacion']))
        p.drawString(550, y, row['calidad_clasificada'])
        
        # Dibujar líneas de división horizontal
        p.line(40, y - 2, width - 40, y - 2)  # Línea horizontal inferior
        y -= 8  # Mover hacia abajo para la siguiente fila

    # Dibujar la última línea inferior de la tabla
    p.line(40, y - 2, width - 40, y - 2)  # Línea final

    p.save()
    buffer.seek(0)
    response.data = buffer.read()
    return response


if __name__ == '__main__':
    app.run(debug=True)
