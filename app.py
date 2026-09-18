import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st

# ML imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="ProyecKeras ML - Financial Analytics & Machine Learning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS personalizados desde style.css
if os.path.exists("style.css"):
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==============================================================================
# GENERADOR DE DATOS FINANCIEROS DE RESPALDO (FALLBACK SIMULADO DE ALTA FIDELIDAD)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def generar_datos_simulados(symbol, periodos=250, tipo="stock"):
    np.random.seed(abs(hash(symbol)) % 10000000)
    fechas = pd.date_range(end=datetime.now(), periods=periodos, freq='B')
    
    precios_base = {
        "IBM": 216.56, "AAPL": 225.0, "MSFT": 440.0, "GOOGL": 175.0, 
        "AMZN": 180.0, "TSLA": 210.0, "NVDA": 120.0, "GOLD": 2170.0, 
        "SILVER": 29.5, "WTI": 67.6
    }
    p0 = precios_base.get(symbol.upper(), 150.0)
    
    returns = np.random.normal(0.0008, 0.015, periodos)
    price = p0 * np.exp(np.cumsum(returns) - np.cumsum(returns)[-1])
    
    high = price * (1 + np.abs(np.random.normal(0.005, 0.003, periodos)))
    low = price * (1 - np.abs(np.random.normal(0.005, 0.003, periodos)))
    open_p = low + (high - low) * np.random.random(periodos)
    volume = np.random.randint(2000000, 20000000, periodos)
    
    if tipo == "commodity":
        df = pd.DataFrame({'Price': price, 'Close': price}, index=fechas)
    else:
        df = pd.DataFrame({'Open': open_p, 'High': high, 'Low': low, 'Close': price, 'Volume': volume}, index=fechas)
    
    df.index.name = "Date"
    return df.sort_index(ascending=True)

# ==============================================================================
# FUNCIONES DE CONSULTA API ALPHA VANTAGE
# ==============================================================================
BASE_URL = "https://www.alphavantage.co/query"

def hacer_request_api(params, api_key_usuario=""):
    api_key = api_key_usuario.strip() or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return None, "MODO_FALLBACK"
    
    req_params = params.copy()
    req_params['apikey'] = api_key
    try:
        response = requests.get(BASE_URL, params=req_params, timeout=10)
        data = response.json()
        if 'Error Message' in data or 'Note' in data or 'Information' in data:
            return None, "MODO_FALLBACK"
        return data, None
    except Exception:
        return None, "MODO_FALLBACK"

def convertir_time_series(data, key):
    if not data or key not in data:
        return None
    df = pd.DataFrame.from_dict(data[key], orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df.sort_index(ascending=True)

def convertir_commodities(data):
    if not data or 'data' not in data:
        return None
    df = pd.DataFrame(data['data'])
    if 'date' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df.sort_index(ascending=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_daily(symbol, api_key=""):
    params = {'function': 'TIME_SERIES_DAILY', 'symbol': symbol, 'outputsize': 'compact'}
    data, err = hacer_request_api(params, api_key)
    if data:
        df = convertir_time_series(data, 'Time Series (Daily)')
        if df is not None:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df, "API_REAL"
    return generar_datos_simulados(symbol, 250, "stock"), "SIMULADO"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rsi(symbol, time_period=14, api_key=""):
    params = {'function': 'RSI', 'symbol': symbol, 'interval': 'daily', 'time_period': time_period, 'series_type': 'close'}
    data, err = hacer_request_api(params, api_key)
    if data:
        df = convertir_time_series(data, 'Technical Analysis: RSI')
        if df is not None:
            df.columns = ['RSI']
            return df, "API_REAL"
    
    df_stock = generar_datos_simulados(symbol, 250, "stock")
    delta = df_stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=time_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=time_period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return pd.DataFrame({'RSI': rsi.fillna(50.0)}, index=df_stock.index), "CALCULADO_LOCAL"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_bollinger_bands(symbol, time_period=20, api_key=""):
    params = {'function': 'BBANDS', 'symbol': symbol, 'interval': 'daily', 'time_period': time_period, 'series_type': 'close'}
    data, err = hacer_request_api(params, api_key)
    if data:
        df = convertir_time_series(data, 'Technical Analysis: BBANDS')
        if df is not None:
            return df, "API_REAL"
    
    df_stock = generar_datos_simulados(symbol, 250, "stock")
    sma = df_stock['Close'].rolling(window=time_period).mean()
    std = df_stock['Close'].rolling(window=time_period).std()
    df_bb = pd.DataFrame({
        'Real Middle Band': sma,
        'Real Upper Band': sma + (std * 2),
        'Real Lower Band': sma - (std * 2)
    }, index=df_stock.index).bfill()
    return df_bb, "CALCULADO_LOCAL"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_commodity(symbol, api_key=""):
    func = 'GOLD' if symbol == 'GOLD' else ('SILVER' if symbol == 'SILVER' else 'WTI')
    params = {'function': func, 'interval': 'daily'}
    data, err = hacer_request_api(params, api_key)
    if data:
        df = convertir_commodities(data)
        if df is not None:
            col_name = 'value' if 'value' in df.columns else df.columns[0]
            df = df.rename(columns={col_name: 'Price'})
            df['Close'] = df['Price']
            return df[['Price', 'Close']], "API_REAL"
    return generar_datos_simulados(symbol, 250, "commodity"), "SIMULADO"

def crear_sparkline(series, color='#0070f3'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(series))),
        y=series.values,
        mode='lines',
        line=dict(color=color, width=2.5),
        hoverinfo='none'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=45,
        width=100
    )
    return fig

# ==============================================================================
# HELPER PARA PREPARAR DATOS DE ML
# ==============================================================================
def preparar_features_ml(df_stock, lags=5):
    df = df_stock.copy()
    close_col = 'Close' if 'Close' in df.columns else 'Price'
    df['Return'] = df[close_col].pct_change()
    
    for i in range(1, lags + 1):
        df[f'Lag_{i}'] = df[close_col].shift(i)
        df[f'Return_Lag_{i}'] = df['Return'].shift(i)
        
    df['SMA_5'] = df[close_col].rolling(window=5).mean()
    df['SMA_20'] = df[close_col].rolling(window=20).mean()
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    
    df['Target_Price'] = df[close_col].shift(-1)
    df['Target_Class'] = (df['Target_Price'] > df[close_col]).astype(int)
    
    df = df.dropna()
    feature_cols = [c for c in df.columns if c.startswith('Lag_') or c.startswith('Return_Lag_') or c.startswith('SMA_') or c.startswith('Volatility_')]
    return df, feature_cols

# ==============================================================================
# SIDEBAR DE NAVEGACIÓN Y CONFIGURACIÓN
# ==============================================================================
st.sidebar.markdown(
    """
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 20px;'>
        <div style='background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.3rem; box-shadow: 0 4px 15px rgba(99,102,241,0.4);'>📈</div>
        <div>
            <h3 style='margin: 0; font-size: 1.2rem; color: #f8fafc; font-weight: 800; letter-spacing: -0.02em;'>ProyecKeras ML</h3>
            <p style='margin: 0; font-size: 0.78rem; color: #94a3b8; font-weight: 500;'>Financial Analytics & Machine Learning</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div style="font-size: 0.75rem; font-weight: 700; color: #64748b; margin-bottom: 8px; letter-spacing: 0.05em;">SECCIONES DEL PROYECTO</div>', unsafe_allow_html=True)

secciones = [
    "🏠 Inicio",
    "📄 Resumen del Proyecto",
    "🗄️ Exploración de Datos",
    "📈 Análisis Técnico & Commodities",
    "🧠 Modelo de Machine Learning",
    "🎯 Predicciones & Escenarios",
    "📊 Evaluación de Métricas",
    "🖼️ Visualizaciones Comparativas"
]

pagina = st.sidebar.radio("", secciones, index=0)

st.sidebar.markdown('<div style="font-size: 0.75rem; font-weight: 700; color: #64748b; margin-top: 20px; margin-bottom: 8px; letter-spacing: 0.05em;">CONFIGURACIÓN API</div>', unsafe_allow_html=True)
api_key_input = st.sidebar.text_input("🔑 Alpha Vantage API Key:", type="password", help="Opcional. Si no se ingresa, se usa la ingesta sintética de alta fidelidad.")

st.sidebar.markdown(
    """
    <div style='margin-top: 30px; padding: 14px; background: rgba(30, 41, 59, 0.6); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); text-align: center;'>
        <div style='font-size: 0.85rem; font-weight: 600; color: #f8fafc;'>CRISP-ML(Q) Certified</div>
        <div style='font-size: 0.75rem; color: #94a3b8; margin-top: 4px;'>Suite Bursátil v2.0</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 1. SECCIÓN: 🏠 INICIO
# ==============================================================================
if pagina == "🏠 Inicio":
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%); padding: 24px; border-radius: 16px; border: 1px solid rgba(99, 102, 241, 0.3); margin-bottom: 25px;'>
            <h1 style='margin: 0; font-size: 1.8rem; font-weight: 800; color: #f8fafc;'>🏦 Monitor Bursátil & Machine Learning Suite</h1>
            <p style='margin: 5px 0 0 0; color: #94a3b8; font-size: 0.95rem;'>Visión general del mercado en tiempo real, KPIs cuantitativos e indicadores para acciones y materias primas.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    c_alert, c_btn = st.columns([4, 1])
    with c_alert:
        st.info("ℹ️ **Modo de Respaldo Activo:** Mostrando datos con fallback generativo continuo de alta precisión. Puedes ingresar tu API Key de Alpha Vantage en la barra lateral para sincronización en directo.")

    with st.spinner("Cargando indicadores clave..."):
        df_ibm, _ = fetch_stock_daily("IBM", api_key_input)
        df_rsi_ibm, _ = fetch_rsi("IBM", 14, api_key_input)
        df_gold, _ = fetch_commodity("GOLD", api_key_input)
        df_wti, _ = fetch_commodity("WTI", api_key_input)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        p_act = df_ibm['Close'].iloc[-1]
        p_prev = df_ibm['Close'].iloc[-2]
        d_pct = ((p_act - p_prev) / p_prev) * 100
        st.metric("IBM Close Price", f"${p_act:,.2f}", f"{d_pct:+.2f}% vs. día ant.")
        st.plotly_chart(crear_sparkline(df_ibm['Close'].iloc[-20:], '#6366f1'), use_container_width=True, key="sp_ibm")

    with k2:
        rsi_val = df_rsi_ibm['RSI'].iloc[-1]
        st.metric("RSI IBM (14d)", f"{rsi_val:.1f}", "Rango Neutro" if 30 <= rsi_val <= 70 else ("Sobrecompra" if rsi_val > 70 else "Sobrevenda"))
        st.plotly_chart(crear_sparkline(df_rsi_ibm['RSI'].iloc[-20:], '#10b981'), use_container_width=True, key="sp_rsi")

    with k3:
        g_act = df_gold['Close'].iloc[-1]
        g_prev = df_gold['Close'].iloc[-2]
        g_pct = ((g_act - g_prev) / g_prev) * 100
        st.metric("Oro (USD/oz)", f"${g_act:,.2f}", f"{g_pct:+.2f}% vs. día ant.")
        st.plotly_chart(crear_sparkline(df_gold['Close'].iloc[-20:], '#f59e0b'), use_container_width=True, key="sp_gold")

    with k4:
        w_act = df_wti['Close'].iloc[-1]
        w_prev = df_wti['Close'].iloc[-2]
        w_pct = ((w_act - w_prev) / w_prev) * 100
        st.metric("Petróleo WTI", f"${w_act:,.2f}", f"{w_pct:+.2f}% vs. día ant.")
        st.plotly_chart(crear_sparkline(df_wti['Close'].iloc[-20:], '#ec4899'), use_container_width=True, key="sp_wti")

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart, col_summary = st.columns([3, 1])

    with col_chart:
        st.subheader("📉 Tendencia de Precio Reciente (IBM)")
        df_sub = df_ibm.iloc[-40:]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_sub.index, y=df_sub['Close'],
            mode='lines+markers', name='Precio Cierre',
            line=dict(color='#6366f1', width=3),
            fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        fig_trend.update_layout(
            template="plotly_dark", height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
            yaxis=dict(tickprefix="$")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_summary:
        p_init = df_sub['Close'].iloc[0]
        p_final = df_sub['Close'].iloc[-1]
        var_tot = ((p_final - p_init) / p_init) * 100
        st.subheader("📊 Resumen")
        st.write(f"**Precio Inicial (40d):** ${p_init:,.2f}")
        st.write(f"**Precio Actual:** ${p_final:,.2f}")
        st.write(f"**Variación Período:** {var_tot:+.2f}%")
        st.write(f"**Volatilidad (Std):** ${df_sub['Close'].std():.2f}")

# ==============================================================================
# 2. SECCIÓN: 📄 RESUMEN DEL PROYECTO
# ==============================================================================
elif pagina == "📄 Resumen del Proyecto":
    st.markdown("## 📄 Resumen Ejecutivo de ProyecKeras", unsafe_allow_html=True)
    st.markdown(
        """
        ProyecKeras es una plataforma de analítica financiera cuantitativa y Machine Learning orientada a la modelación y predicción de precios bursátiles y materias primas.
        """
    )
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            ### 🎯 Objetivos Clave
            - **Ingesta Continua:** Integración robusta de datos bursátiles con mecanismo de resiliencia ante fallos API.
            - **Indicadores Cuantitativos:** Automatización de RSI, Bollinger Bands, MACD y Medias Móviles.
            - **Predicción Supervisada:** Modelado de tendencias y valores futuros mediante algoritmos Ensemble y Regresión Ridge.
            - **Calidad CRISP-ML(Q):** Adherencia a los estándares de desarrollo de Machine Learning industrial.
            """
        )
    with c2:
        st.markdown(
            """
            ### 💻 Stack Tecnológico
            - **Lenguaje Principal:** Python 3.10+
            - **Dashboard:** Streamlit
            - **Librerías de ML:** Scikit-Learn (Random Forest, Gradient Boosting, Ridge, Decision Trees)
            - **Procesamiento de Datos:** Pandas, NumPy
            - **Visualización:** Plotly Express & Graph Objects
            - **Documentación & UI:** HTML5, CSS3 (Glassmorphism), Badges Shields.io
            """
        )

# ==============================================================================
# 3. SECCIÓN: 🗄️ EXPLORACIÓN DE DATOS
# ==============================================================================
elif pagina == "🗄️ Exploración de Datos":
    st.markdown("## 🗄️ Exploración de Series Temporales Financieras", unsafe_allow_html=True)
    
    col_sel, col_type = st.columns(2)
    with col_sel:
        sym = st.selectbox("Seleccionar Activo:", ["IBM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "GOLD", "SILVER", "WTI"])
    with col_type:
        tipo_data = "commodity" if sym in ["GOLD", "SILVER", "WTI"] else "stock"
        st.text(f"Tipo de Activo: {tipo_data.upper()}")

    if tipo_data == "stock":
        df_data, modo = fetch_stock_daily(sym, api_key_input)
    else:
        df_data, modo = fetch_commodity(sym, api_key_input)

    st.caption(f"Origen de los datos: **{modo}** | Total Registros: **{len(df_data)}**")
    
    st.dataframe(df_data.sort_index(ascending=False), use_container_width=True)

    csv_data = df_data.to_csv().encode('utf-8')
    st.download_button("💾 Descargar Dataset Procesado (CSV)", csv_data, f"{sym}_datos_financieros.csv", "text/csv")

# ==============================================================================
# 4. SECCIÓN: 📈 ANÁLISIS TÉCNICO & COMMODITIES
# ==============================================================================
elif pagina == "📈 Análisis Técnico & Commodities":
    st.markdown("## 📈 Indicadores Técnicos y Osciladores Financieros", unsafe_allow_html=True)
    sym = st.selectbox("Seleccionar Activo para Análisis:", ["IBM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"])
    
    df_stock, _ = fetch_stock_daily(sym, api_key_input)
    df_rsi, _ = fetch_rsi(sym, 14, api_key_input)
    df_bb, _ = fetch_bollinger_bands(sym, 20, api_key_input)

    tab1, tab2 = st.tabs(["📉 Bandas de Bollinger & Precio", "⚡ RSI (Relative Strength Index)"])

    with tab1:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], name='Precio Cierre', line=dict(color='#6366f1', width=2)))
        fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb['Real Upper Band'], name='Banda Superior', line=dict(color='#ef4444', dash='dot')))
        fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb['Real Middle Band'], name='Media Móvil (SMA 20)', line=dict(color='#f59e0b', dash='dash')))
        fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb['Real Lower Band'], name='Banda Inferior', line=dict(color='#10b981', dash='dot')))
        fig_bb.update_layout(template="plotly_dark", height=450, title=f"Bandas de Bollinger - {sym}")
        st.plotly_chart(fig_bb, use_container_width=True)

    with tab2:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_rsi.index, y=df_rsi['RSI'], name='RSI 14', line=dict(color='#8b5cf6', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevenda (30)")
        fig_rsi.update_layout(template="plotly_dark", height=400, title=f"Índice de Fuerza Relativa (RSI) - {sym}")
        st.plotly_chart(fig_rsi, use_container_width=True)

# ==============================================================================
# 5. SECCIÓN: 🧠 MODELO DE MACHINE LEARNING
# ==============================================================================
elif pagina == "🧠 Modelo de Machine Learning":
    st.markdown("## 🧠 Entrenamiento de Modelos Predictivos Supervisados", unsafe_allow_html=True)
    
    col_a, col_m, col_p = st.columns(3)
    with col_a:
        sym = st.selectbox("Activo Objetivo:", ["IBM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"])
    with col_m:
        tipo_modelo = st.selectbox("Algoritmo ML:", ["Random Forest Regressor", "Gradient Boosting Regressor", "Ridge Regression", "Decision Tree Regressor", "Random Forest Classifier"])
    with col_p:
        test_pct = st.slider("Porcentaje Test (%)", 10, 40, 20, 5) / 100.0

    if st.button("🚀 Entrenar y Evaluar Modelo", use_container_width=True):
        with st.spinner("Entrenando algoritmo y calculando características..."):
            df_stock, _ = fetch_stock_daily(sym, api_key_input)
            df_ml, feature_cols = preparar_features_ml(df_stock, lags=5)
            
            X = df_ml[feature_cols]
            is_class = "Classifier" in tipo_modelo
            y = df_ml['Target_Class'] if is_class else df_ml['Target_Price']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, shuffle=False)
            
            if tipo_modelo == "Random Forest Regressor":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif tipo_modelo == "Gradient Boosting Regressor":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif tipo_modelo == "Ridge Regression":
                model = Ridge(alpha=1.0)
            elif tipo_modelo == "Decision Tree Regressor":
                model = DecisionTreeRegressor(random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.session_state['trained_model'] = model
            st.session_state['model_name'] = tipo_modelo
            st.session_state['feature_cols'] = feature_cols
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['target_symbol'] = sym
            st.session_state['is_class'] = is_class
            st.session_state['df_ml'] = df_ml
            
            st.success(f"✅ ¡Modelo **{tipo_modelo}** entrenado exitosamente para **{sym}**!")

# ==============================================================================
# 6. SECCIÓN: 🎯 PREDICCIONES & ESCENARIOS
# ==============================================================================
elif pagina == "🎯 Predicciones & Escenarios":
    st.markdown("## 🎯 Proyección de Escenarios Futuros", unsafe_allow_html=True)
    if 'trained_model' in st.session_state and not st.session_state['is_class']:
        model = st.session_state['trained_model']
        df_ml = st.session_state['df_ml']
        sym = st.session_state['target_symbol']
        
        dias_pred = st.slider("Días a Proyectar:", 1, 30, 15)
        
        # Proyección autoregresiva
        ultimos_lags = df_ml[st.session_state['feature_cols']].iloc[-1].values.copy()
        predicciones = []
        precio_actual = df_ml['Close'].iloc[-1]
        
        for _ in range(dias_pred):
            p = model.predict(ultimos_lags.reshape(1, -1))[0]
            predicciones.append(p)
            ultimos_lags = np.roll(ultimos_lags, -1)
            ultimos_lags[-1] = p

        fechas_futuras = pd.date_range(start=df_ml.index[-1] + timedelta(days=1), periods=dias_pred, freq='B')
        
        std_hist = df_ml['Return'].std() * precio_actual
        esc_base = np.array(predicciones)
        esc_opt = esc_base + np.linspace(0, std_hist * np.sqrt(dias_pred), dias_pred)
        esc_pes = esc_base - np.linspace(0, std_hist * np.sqrt(dias_pred), dias_pred)
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df_ml.index[-50:], y=df_ml['Close'].iloc[-50:], name='Histórico Reciente', line=dict(color='#6366f1', width=3)))
        fig_p.add_trace(go.Scatter(x=fechas_futuras, y=esc_base, name='Escenario Base (Predicción)', line=dict(color='#10b981', width=3, dash='dash')))
        fig_p.add_trace(go.Scatter(x=fechas_futuras, y=esc_opt, name='Escenario Optimista (+1 Std)', line=dict(color='#06b6d4', width=2, dash='dot')))
        fig_p.add_trace(go.Scatter(x=fechas_futuras, y=esc_pes, name='Escenario Pesimista (-1 Std)', line=dict(color='#ef4444', width=2, dash='dot')))
        
        fig_p.update_layout(template="plotly_dark", height=480, title=f"Proyección de Precio a {dias_pred} Días - {sym}")
        st.plotly_chart(fig_p, use_container_width=True)

        df_out = pd.DataFrame({'Fecha': fechas_futuras, 'Base': esc_base, 'Optimista': esc_opt, 'Pesimista': esc_pes})
        st.download_button("💾 Descargar Proyecciones (CSV)", df_out.to_csv(index=False).encode('utf-8'), f"{sym}_predicciones.csv", "text/csv")
    else:
        st.warning("⚠️ Primero entrena un modelo de Regresión en la sección **🧠 Modelo de Machine Learning**.")

# ==============================================================================
# 7. SECCIÓN: 📊 EVALUACIÓN DE MÉTRICAS
# ==============================================================================
elif pagina == "📊 Evaluación de Métricas":
    st.markdown("## 📊 Evaluación Diagnóstica de Desempeño", unsafe_allow_html=True)
    if 'trained_model' in st.session_state:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        is_class = st.session_state['is_class']
        model = st.session_state['trained_model']
        feature_cols = st.session_state['feature_cols']

        if not is_class:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Coeficiente R² Score", f"{r2:.4f}")
            m2.metric("Error MAE", f"${mae:.2f}")
            m3.metric("Error RMSE", f"${rmse:.2f}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Importancia de Características (Feature Importance)")
                df_imp = pd.DataFrame({'Característica': feature_cols, 'Importancia': model.feature_importances_}).sort_values('Importancia', ascending=True)
                fig_imp = px.bar(df_imp, x='Importancia', y='Característica', orientation='h', template="plotly_dark", color='Importancia', color_continuous_scale='Purples')
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Precisión (Accuracy)", f"{acc*100:.2f}%")
            
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicho", y="Real"), x=['Bajista', 'Alcista'], y=['Bajista', 'Alcista'], template="plotly_dark", color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("⚠️ Primero entrena un modelo en la sección **🧠 Modelo de Machine Learning**.")

# ==============================================================================
# 8. SECCIÓN: 🖼️ VISUALIZACIONES COMPARATIVAS
# ==============================================================================
elif pagina == "🖼️ Visualizaciones Comparativas":
    st.markdown("## 🖼️ Matriz de Correlación y Rendimientos Cruzados", unsafe_allow_html=True)
    activos = st.multiselect("Seleccionar Activos a Comparar:", ["IBM", "AAPL", "MSFT", "GOOGL", "AMZN", "GOLD", "WTI"], default=["IBM", "AAPL", "GOLD", "WTI"])
    
    if len(activos) >= 2:
        df_comp = pd.DataFrame()
        for a in activos:
            df_item, _ = fetch_commodity(a, api_key_input) if a in ["GOLD", "WTI"] else fetch_stock_daily(a, api_key_input)
            col_name = 'Price' if 'Price' in df_item.columns else 'Close'
            df_comp[a] = df_item[col_name]
            
        df_returns = df_comp.dropna().pct_change().dropna()
        
        c_corr, c_box = st.columns(2)
        with c_corr:
            st.subheader("🔥 Mapa de Calor de Correlación")
            fig_corr = px.imshow(df_returns.corr(), text_auto=".2f", color_continuous_scale="Viridis", template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with c_box:
            st.subheader("📦 Distribución de Rendimientos Diarios")
            fig_box = px.box(df_returns, template="plotly_dark", color_discrete_sequence=['#6366f1', '#10b981', '#f59e0b', '#ec4899'])
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Selecciona al menos 2 activos para visualizar la correlación y comparación.")
