import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard de Inversión Pública: Reporte SNIP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }
    .alert-box {
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data_por_año.csv')
    df['year'] = df['year'].astype(int)
    df['especie'] = df['especie'].astype(str)
    df['Calculation'] = df['Calculation'].fillna("Sin Datos")
    if 'departamento' not in df.columns and 'ubicacion_geografica' in df.columns:
        df['departamento'] = df['ubicacion_geografica'].str.split(',').str[-1].str.strip()
    # Execution rates per row
    df['tasa_ejec_fisica']       = np.where(df['meta_fisica']      > 0, df['meta_ejecutada']   / df['meta_fisica'],      np.nan)
    df['tasa_ejec_presupuestal'] = np.where(df['monto_solicitado'] > 0, df['monto_ejecutado']  / df['monto_solicitado'], np.nan)
    return df

@st.cache_data
def load_guatecompras():
    return pd.read_csv('snip_guatecompras_completo2.csv')

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def calc_unit_cost(monto_ej, meta_ej, monto_sol, meta_fis):
    if monto_ej > 0 and meta_ej > 0:
        return monto_ej / meta_ej
    elif monto_sol > 0 and meta_fis > 0:
        return monto_sol / meta_fis
    return np.nan

def add_zscore(df, col, new_col='z_score'):
    mu, sigma = df[col].mean(), df[col].std()
    df[new_col] = (df[col] - mu) / sigma if sigma > 0 else 0.0
    return df

# true or false column 
def add_iqr_flag(df, col, new_col='iqr_flag'):
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    df[new_col] = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    return df

def style_outliers(row):
    z = row.get('z_score', 0) or 0
    if abs(z) > 2.5:
        return ['background-color: #fca5a5'] * len(row)  # strong red
    if abs(z) > 1.5:
        return ['background-color: #fed7aa'] * len(row)  # orange
    return [''] * len(row)

def risk_badge(z):
    if z > 2.5:   return "🔴 Alto"
    if z > 1.5:   return "🟡 Moderado"
    return "🟢 Normal"

COL_PROVEEDOR = 'compiledRelease/awards/0/suppliers/0/name'
COL_ADJUDICADO = 'compiledRelease/awards/0/value/amount'
COL_NOG = 'compiledRelease/tender/id'

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
try:
    df_raw = load_data()
    df_gc  = load_guatecompras()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
st.sidebar.header("Filtros Globales")

all_etapas = sorted(df_raw['etapa_actual'].dropna().unique())
selected_etapas = st.sidebar.multiselect("Etapa Actual", all_etapas, default=all_etapas)

all_especies = sorted(df_raw['especie'].unique())
default_selection = ["CALLE"] if "CALLE" in all_especies else ([all_especies[0]] if all_especies else [])
selected_especies = st.sidebar.multiselect("Especie", all_especies,
    default=default_selection)

if selected_especies:
    available_pt = sorted(df_raw[df_raw['especie'].isin(selected_especies)]['product_type'].dropna().unique())
else:
    available_pt = sorted(df_raw['product_type'].dropna().unique())
selected_product_type = st.sidebar.selectbox("Tipo de Producto", options=available_pt,
    index=0 if available_pt else None)

all_departamentos = sorted(df_raw['departamento'].dropna().unique())
selected_departamentos = st.sidebar.multiselect("Departamento", all_departamentos, default=all_departamentos)

all_sectores = sorted(df_raw['sector'].dropna().unique())
selected_sectores = st.sidebar.multiselect("Sectores", all_sectores, default=all_sectores)

all_instituciones = sorted(df_raw['institucion'].dropna().unique()) if 'institucion' in df_raw.columns else []
selected_instituciones = st.sidebar.multiselect("Institución", all_instituciones, default=all_instituciones)

# ─────────────────────────────────────────────
# GLOBAL FILTERED DATAFRAMES  (computed once)
# ─────────────────────────────────────────────
base_mask = (
    df_raw['especie'].isin(selected_especies) &
    df_raw['departamento'].isin(selected_departamentos) &
    df_raw['etapa_actual'].isin(selected_etapas) &
    df_raw['sector'].isin(selected_sectores) &
    (df_raw['product_type'] == selected_product_type)
)
if all_instituciones:
    base_mask = base_mask & df_raw['institucion'].isin(selected_instituciones)

df_filtered = df_raw[base_mask].copy()

# ── Yearly view ──────────────────────────────
df_yearly = df_filtered.dropna(subset=['cost_per_unit']).copy()
if not df_yearly.empty:
    df_yearly = add_zscore(df_yearly, 'cost_per_unit')
    df_yearly = add_iqr_flag(df_yearly, 'cost_per_unit')

# ── Accumulated view (one row per SNIP) ──────
agg_dict = {
    'monto_ejecutado': 'sum', 'meta_ejecutada': 'sum',
    'monto_solicitado': 'sum', 'meta_fisica': 'sum',
    'especie': 'first', 'departamento': 'first',
    'sector': 'first', 'latitud': 'first', 'longitud': 'first',
    'etapa_actual': 'first',
    'tasa_ejec_fisica': 'mean', 'tasa_ejec_presupuestal': 'mean',
}
if 'institucion' in df_filtered.columns:
    agg_dict['institucion'] = 'first'
if 'unidad_ejecutora' in df_filtered.columns:
    agg_dict['unidad_ejecutora'] = 'first'

df_acumulado = df_filtered.groupby('snip').agg(agg_dict).reset_index()
df_acumulado['cost_total_por_unidad'] = df_acumulado.apply(
    lambda r: calc_unit_cost(r['monto_ejecutado'], r['meta_ejecutada'],
                             r['monto_solicitado'], r['meta_fisica']), axis=1)

plot_total = df_acumulado.dropna(subset=['cost_total_por_unidad']).copy()
if not plot_total.empty:
    plot_total = add_zscore(plot_total, 'cost_total_por_unidad')
    plot_total = add_iqr_flag(plot_total, 'cost_total_por_unidad')
    plot_total['riesgo'] = plot_total['z_score'].apply(risk_badge)

# ── Guatecompras merge ────────────────────────
df_consolidado = df_acumulado.dropna(subset=['cost_total_por_unidad']).copy()
_gc_cols = ['snip', COL_PROVEEDOR, COL_ADJUDICADO, COL_NOG]
_gc_avail = [c for c in _gc_cols if c in df_gc.columns]
df_prov = df_consolidado.merge(df_gc[_gc_avail], on='snip', how='inner')
if not df_prov.empty and COL_ADJUDICADO in df_prov.columns:
    df_prov['diferencia'] = df_prov[COL_ADJUDICADO] - df_prov['monto_ejecutado']
    df_prov = add_zscore(df_prov, 'cost_total_por_unidad')

# ── YoY cost change ───────────────────────────
df_yoy = (
    df_filtered
    .dropna(subset=['cost_per_unit'])
    .sort_values(['snip', 'year'])
    .copy()
)
df_yoy['cost_prev'] = df_yoy.groupby('snip')['cost_per_unit'].shift(1)
df_yoy['yoy_pct']   = (df_yoy['cost_per_unit'] - df_yoy['cost_prev']) / df_yoy['cost_prev'] * 100
df_yoy = df_yoy.dropna(subset=['yoy_pct'])

# ─────────────────────────────────────────────
# DASHBOARD HEADER KPIs
# ─────────────────────────────────────────────
st.title("Dashboard de Inversión Pública — SNIP")

if not plot_total.empty:
    n_total    = len(plot_total)
    n_alert    = int((plot_total['z_score'] > 1.5).sum())
    n_iqr      = int(plot_total['iqr_flag'].sum())
    med_costo  = plot_total['cost_total_por_unidad'].median()
    prod_costo  = plot_total['cost_total_por_unidad'].mean()
    std_costo  = plot_total['cost_total_por_unidad'].std()
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Proyectos", f"{n_total:,}")
    c2.metric("Alertas Z > 1.5", f"{n_alert:,}", delta=f"{n_alert/n_total*100:.1f}% del total", delta_color="inverse")
    c3.metric("Alertas IQR", f"{n_iqr:,}", delta=f"{n_iqr/n_total*100:.1f}% del total", delta_color="inverse")
    c4.metric("Costo Mediano por Unidad (Q)", f"Q{med_costo:,.0f}")
    c5.metric("Costo Promedio por Unidad (Q)", f"Q{prod_costo:,.0f}")
    c6.metric("Desv. Estandar", f"Q{std_costo:,.0f}")

    st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_yearly, tab_acumulado, tab_correlation, tab_guatecompras, tab_invest = st.tabs([
    "Vista Anual",
    "Por Proyecto",
    "Regional",
    "Guatecompras",
    "Investigar SNIP",
])

# ══════════════════════════════════════════════
# TAB 1 — VISTA ANUAL
# ══════════════════════════════════════════════
with tab_yearly:
    st.header("Costos Unitarios por Ejercicio Fiscal")
    st.info("Variabilidad de precios según el ejercicio fiscal. Escala logarítmica: diferencias visuales pequeñas representan grandes diferencias reales.")

    with st.expander("Filtrar años"):
        all_years = sorted(df_raw['year'].unique())
        selected_years = st.multiselect("Años", all_years, default=all_years)

    df_yr = df_yearly[df_yearly['year'].isin(selected_years)].copy()

    if df_yr.empty:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        med_order_depto = (
            df_yr.groupby("departamento")["cost_per_unit"]
            .median().sort_values(ascending=False).index.tolist()
        )


        fig = px.box(df_yr, x="departamento", y="cost_per_unit", color="especie",
                hover_data=["snip"], log_y=True, template="plotly_white",
                title="Costos por Unidad — por Departamento",
                category_orders={"departamento": med_order_depto})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df_yr, x="year", y="cost_per_unit", color="especie",
                hover_data=["snip"], log_y=True, template="plotly_white",
                title="Costos por Unidad — por Año")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Resumen por Departamento")
        stats_d = df_yr.groupby('departamento')['cost_per_unit'].agg(
            ['count','mean','median','std','min','max']).reset_index()
        stats_d.columns = ['Departamento','N°','Promedio (Q)','Mediana (Q)','Desv. Std','Mínimo (Q)','Máximo (Q)']
        st.dataframe(stats_d.style.format(precision=2, thousands=","), use_container_width=True)

        st.subheader("Resumen por Sector")
        stats_s = df_yr.groupby('sector')['cost_per_unit'].agg(
            ['count','mean','median','std','min','max']).reset_index()
        stats_s.columns = ['Sector','N°','Promedio (Q)','Mediana (Q)','Desv. Std','Mínimo (Q)','Máximo (Q)']
        st.dataframe(stats_s.style.format(precision=2, thousands=","), use_container_width=True)

        st.divider()
        st.subheader("Detalle de Registros")
        st.caption("🟥 Z > 2.5 (alto)   🟧 Z > 1.5 (moderado)")
        show_cols = [c for c in ['snip','year','cost_per_unit','z_score','iqr_flag',
            'monto_ejecutado','meta_ejecutada','monto_solicitado','meta_fisica',
            'tasa_ejec_fisica','tasa_ejec_presupuestal','Calculation',
            'etapa_actual','resultado_historial','departamento','unidad_ejecutora','especie','sector']
            if c in df_yr.columns]
        
        column_labels = {
            'snip':                    'Código SNIP',
            'year':                    'Año',
            'cost_per_unit':           'Costo por Unidad (Q)',
            'z_score':                 'z_score',
            'iqr_flag':                'Alerta IQR',
            'monto_ejecutado':         'Monto Ejecutado (Q)',
            'meta_ejecutada':          'Meta Ejecutada (unidades)',
            'monto_solicitado':        'Monto Solicitado (Q)',
            'meta_fisica':             'Meta Física (unidades)',
            'tasa_ejec_fisica':        'Ejec. Física (%)',
            'tasa_ejec_presupuestal':  'Ejec. Presupuestal (%)',
            'Calculation':             'Fuente del Cálculo',
            'etapa_actual':            'Etapa Actual',
            'resultado_historial':     'Resultado Historial',
            'departamento':            'Departamento',
            'unidad_ejecutora':        'Unidad Ejecutora',
            'especie':                 'Especie',
            'sector':                  'Sector',
        }
        
        st.dataframe(
            df_yr.sort_values('z_score', ascending=False)[show_cols].rename(columns=column_labels)
            .style.apply(style_outliers, axis=1).format(precision=2, thousands=","),
            use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — POR PROYECTO (ACUMULADO)
# ══════════════════════════════════════════════
with tab_acumulado:
    st.header("Análisis Consolidado por Ciclo de Proyecto")
    st.info("Unifica todos los registros históricos de cada SNIP para determinar el Costo Unitario Final.")

    if plot_total.empty:
        st.warning("No hay datos suficientes.")
    else:
        med_order = (
            plot_total.groupby("departamento")["cost_total_por_unidad"]
            .median().sort_values(ascending=False).index.tolist()
        )
        fig = px.box(plot_total, x="departamento", y="cost_total_por_unidad",
            color="especie", log_y=True, hover_data=["snip","riesgo"],
            template="plotly_white",
            title="Costo Total por Unidad — por Departamento",
            category_orders={"departamento": med_order})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("Por Departamento")
        s = plot_total.groupby('departamento').agg(
                Proyectos=('snip','count'),
                Promedio=('cost_total_por_unidad','mean'),
                Mediana=('cost_total_por_unidad','median'),
                Alertas=('z_score', lambda x: (x>1.5).sum())
            ).reset_index().sort_values('Alertas', ascending=False)
        st.dataframe(s.style.format(precision=2, thousands=","), use_container_width=True)


        st.subheader("Por Sector")
        s2 = plot_total.groupby('sector').agg(
                Proyectos=('snip','count'),
                Promedio=('cost_total_por_unidad','mean'),
                Mediana=('cost_total_por_unidad','median'),
                Alertas=('z_score', lambda x: (x>1.5).sum())
            ).reset_index().sort_values('Alertas', ascending=False)
        st.dataframe(s2.style.format(precision=2, thousands=","), use_container_width=True)

        st.divider()
        st.subheader("Detalle de Proyectos")
        
        # Add proveedor 
        if not df_prov.empty and COL_PROVEEDOR in df_prov.columns:
            prov_per_snip = df_prov[['snip', COL_PROVEEDOR]].rename(
                columns={COL_PROVEEDOR: 'proveedor'}
            )
            plot_total_display = plot_total.merge(prov_per_snip, on='snip', how='left')
        else:
            plot_total_display = plot_total.copy()


        st.caption("🟥 Z > 2.5 (alto)   🟧 Z > 1.5 (moderado)")
        show_cols2 = [c for c in ['snip','cost_total_por_unidad','z_score','iqr_flag',
            'monto_ejecutado','meta_ejecutada','monto_solicitado','meta_fisica',
            'tasa_ejec_fisica','tasa_ejec_presupuestal','etapa_actual',
            'departamento','especie','sector', 'proveedor'] if c in plot_total_display.columns]
        st.dataframe(
            plot_total_display.sort_values('z_score', ascending=False)[show_cols2]
            .style.apply(style_outliers, axis=1).format(precision=2, thousands=","),
            use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — REGIONAL / CORRELACIÓN
# ══════════════════════════════════════════════
with tab_correlation:
    st.header("Comparativa Regional")

    if plot_total.empty:
        st.warning("No hay datos.")
    else:
        # Scatter: monto vs meta
        df_corr = plot_total.copy()
        df_corr['monto_plot'] = np.where(df_corr['monto_ejecutado'] > 0,
            df_corr['monto_ejecutado'], df_corr['monto_solicitado'])
        df_corr['meta_plot'] = np.where(df_corr['meta_ejecutada'] > 0,
            df_corr['meta_ejecutada'], df_corr['meta_fisica'])
        df_corr = df_corr.dropna(subset=['monto_plot','meta_plot'])

        fig_sc = px.scatter(df_corr, x='meta_plot', y='monto_plot',
            color='riesgo',
            color_discrete_map={'🔴 Alto':'#ef4444','🟡 Moderado':'#f59e0b','🟢 Normal':'#22c55e'},
            size='monto_plot', hover_data={'snip':True,'departamento':True,
                'monto_plot':':,.2f','meta_plot':':,.2f'},
            template='plotly_white', log_x=True, log_y=True,
            title='Correlación Monto vs. Meta Física (por SNIP — escala logarítmica)',
            labels={'meta_plot':'Unidades Físicas Totales','monto_plot':'Monto Total (Q)'})
        st.plotly_chart(fig_sc, use_container_width=True)

 
        # Regional deviation
        avg_nac = plot_total['cost_total_por_unidad'].mean()
        df_disp = plot_total.groupby('departamento')['cost_total_por_unidad'].mean().reset_index()
        df_disp['desviacion_pct'] = (df_disp['cost_total_por_unidad'] - avg_nac) / avg_nac * 100
        fig_d = px.bar(df_disp.sort_values('desviacion_pct'), x='desviacion_pct', y='departamento',
                orientation='h', color='desviacion_pct', color_continuous_scale='RdYlGn_r',
                template='plotly_white',
                title=f'Desviación % vs. Promedio Nacional (Q{avg_nac:,.0f})',
                labels={'desviacion_pct':'Diferencia %'})
        fig_d.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_d, use_container_width=True)

        # Map
        df_mapa = plot_total.copy()
        df_mapa['latitud']  = pd.to_numeric(df_mapa['latitud'], errors='coerce')
        df_mapa['longitud'] = pd.to_numeric(df_mapa['longitud'], errors='coerce')
        df_mapa = df_mapa.dropna(subset=['latitud','longitud'])
        df_mapa['inversion'] = np.where(df_mapa['monto_ejecutado'] > 0,
                df_mapa['monto_ejecutado'], df_mapa['monto_solicitado'])
        if not df_mapa.empty:
            fig_map = px.scatter_mapbox(df_mapa,
                lat='latitud', lon='longitud',
                color='z_score', size='inversion',
                color_continuous_scale='RdYlGn_r', size_max=15, zoom=6.5,
                hover_name='snip',
                hover_data={'especie':True,'cost_total_por_unidad':':,.2f',
                        'inversion':':,.2f','departamento':True,
                        'latitud':False,'longitud':False,'z_score':True},
                    mapbox_style='carto-positron',
                    title='Geolocalización (color: Z-Score | tamaño: inversión)')
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No hay coordenadas válidas disponibles.")

# ══════════════════════════════════════════════
# TAB 4 — GUATECOMPRAS
# ══════════════════════════════════════════════
with tab_guatecompras:
    st.header("Integración SNIP + Guatecompras")

    if df_prov.empty:
        st.warning("No hay proyectos con registros en Guatecompras para los filtros actuales.")
    else:
        # Provider concentration alert
        if COL_PROVEEDOR in df_prov.columns:
            df_conc = (
                df_prov.groupby(['departamento', COL_PROVEEDOR])['monto_ejecutado'].sum()
                .reset_index()
            )
            total_dept = df_conc.groupby('departamento')['monto_ejecutado'].sum()
            df_conc['pct'] = df_conc.apply(
                lambda r: r['monto_ejecutado'] / total_dept[r['departamento']] * 100
                if total_dept[r['departamento']] > 0 else 0, axis=1)
            high_conc = df_conc[df_conc['pct'] > 50]
            if not high_conc.empty:
                st.error(f"🔴 **{len(high_conc)} alerta(s) de concentración de mercado:** un proveedor controla >50% del monto en un departamento.")
                with st.expander("Ver alertas de concentración"):
                    st.dataframe(high_conc.sort_values('pct', ascending=False).rename(columns=
                                {COL_PROVEEDOR:'Proveedor', 'monto_ejecutado': 'Monto Ejecutado (Q)','pct':'Concentración (%)'})
                        .style.format({'Monto Ejecutado (Q)':'{:,.0f}','Concentración (%)':'{:.1f}%'}),
                        use_container_width=True)

        # Summary by institution
        if 'institucion' in df_prov.columns and COL_PROVEEDOR in df_prov.columns:
            st.subheader("Resumen por Institución y Unidad Ejecutora")
            df_inst = df_prov.groupby(['institucion','unidad_ejecutora'] if 'unidad_ejecutora' in df_prov.columns else ['institucion']).agg(
                Proyectos=('snip','count'),
                Monto_Ejecutado=('monto_ejecutado','sum'),
            ).reset_index().sort_values('Monto_Ejecutado', ascending=False)
            st.dataframe(df_inst.rename(columns = {"Monto_Ejecutado": 'Monto Ejecutado (Q)'}).style.format({'Monto Ejecutado (Q)':'{:,.0f}'}), use_container_width=True)

        # Top providers bar chart
        if COL_PROVEEDOR in df_prov.columns:
            st.subheader("Top 15 Proveedores por Monto Ejecutado")
            df_prov_top = (
                df_prov.groupby(COL_PROVEEDOR)['monto_ejecutado'].sum()
                .reset_index().nlargest(15, 'monto_ejecutado')
                .sort_values('monto_ejecutado')
            )
            fig_prov = px.bar(df_prov_top, x='monto_ejecutado', y=COL_PROVEEDOR,
                orientation='h', template='plotly_white',
                color='monto_ejecutado', color_continuous_scale='Blues',
                title='Top 15 Proveedores — Monto Ejecutado Total',
                labels={'monto_ejecutado':'Monto Ejecutado (Q)', COL_PROVEEDOR : 'Proveedor'}, 
                hover_data={'monto_ejecutado':':,.0f'})
            fig_prov.update_layout(coloraxis_showscale=False, yaxis_title='')
            st.plotly_chart(fig_prov, use_container_width=True)

            # Provider performance table
            st.subheader("Desempeño Consolidado por Proveedor")
            prov_stats = df_prov.groupby(COL_PROVEEDOR).agg(
                Proyectos=('snip','count'),
                Monto_Ejecutado=('monto_ejecutado','sum'),
                Monto_Adjudicado=(COL_ADJUDICADO,'sum') if COL_ADJUDICADO in df_prov.columns else ('monto_ejecutado','sum'),
                Meta_Ejecutada=('meta_ejecutada','sum'),
                Costo_Prom_Unitario=('cost_total_por_unidad','mean'),
                Proyectos_Alerta=('z_score', lambda x: (x > 1.5).sum()),
            ).reset_index().sort_values('Monto_Ejecutado', ascending=False)
            if COL_ADJUDICADO in df_prov.columns:
                prov_stats['No_Ejecutado'] = prov_stats['Monto_Adjudicado'] - prov_stats['Monto_Ejecutado']

            fmt_p = {c: '{:,.0f}' for c in prov_stats.select_dtypes('number').columns}
            st.dataframe(prov_stats.style.format(fmt_p, thousands=","),
                use_container_width=True)

            # Provider drilldown
            st.divider()
            st.subheader("Explorador por Proveedor")
            prov_list = sorted(df_prov[COL_PROVEEDOR].dropna().unique().astype(str))
            prov_sel = st.selectbox("Selecciona un proveedor:", prov_list,
                index=None, placeholder="Escribe o selecciona...")
            if prov_sel:
                df_pv = df_prov[df_prov[COL_PROVEEDOR] == prov_sel].copy()
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Proyectos", len(df_pv))
                p2.metric("Monto Ejecutado", f"Q{df_pv['monto_ejecutado'].sum():,.0f}")
                p3.metric("Proyectos con Alerta", int((df_pv['z_score'] > 1.5).sum()))
                if COL_ADJUDICADO in df_pv.columns:
                    p4.metric("No Ejecutado (Q)", f"Q{(df_pv[COL_ADJUDICADO] - df_pv['monto_ejecutado']).sum():,.0f}")

                pv_cols = [c for c in ['snip', COL_NOG, 'departamento', 'sector',
                    'monto_ejecutado','monto_solicitado','cost_total_por_unidad',
                    COL_ADJUDICADO,'diferencia', 'etapa_actual', 'z_score'] if c in df_pv.columns]
                st.dataframe(
                    df_pv[pv_cols].sort_values('z_score', ascending=False).rename(columns = {
                        COL_NOG : 'NOG', 'monto_ejecutado': 'Monto Ejecutado (Q)', 'monto_solicitado': 'Monto Solicitado (Q)',
                        COL_ADJUDICADO : 'Monto Adjudicado (Q)', 'diferencia': 'Monto Adjudicado No Ejecutado (Q)', 
                        'costo_total_por_unidad' : 'Costo Por Unidad'})
                    .style.apply(style_outliers, axis=1).format(precision=2, thousands=","),
                    use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — INVESTIGAR SNIP
# ══════════════════════════════════════════════
with tab_invest:
    st.header("Investigador de Proyectos")
    st.info("Selecciona un SNIP para ver su historia completa, proveedor, y señales de alerta.")

    if plot_total.empty:
        st.warning("No hay datos disponibles.")
    else:
        # Build selector labels
        df_sel = plot_total.sort_values('z_score', ascending=False).copy()
        df_sel['_label'] = df_sel.apply(
            lambda r: f"{r['snip']}  |  {r['departamento']}  |  Z: {r['z_score']:.2f}  {risk_badge(r['z_score'])}",
            axis=1)
        label_map = dict(zip(df_sel['_label'], df_sel['snip']))

        col_s1, col_s2 = st.columns([3,1])
        with col_s2:
            only_alerts = st.checkbox("Solo alertas (Z > 1.5)")
        options = df_sel[df_sel['z_score'] > 1.5]['_label'].tolist() if only_alerts else df_sel['_label'].tolist()
        with col_s1:
            sel_label = st.selectbox("Proyecto a investigar (ordenado por Z-Score):",
                options, index=None, placeholder="Escribe o elige un SNIP...")

        if not sel_label:
            st.info("Selecciona un proyecto para comenzar la investigación.")
        elif sel_label in label_map:
            snip_id = label_map[sel_label]
            acc = plot_total[plot_total['snip'] == snip_id].iloc[0]
            z, depto, sector_s = acc['z_score'], acc['departamento'], acc.get('sector','N/D')
            costo_u = acc['cost_total_por_unidad']

            # Alert banner
            if z > 2.5:
                st.error(f"🔴 ALERTA ALTA — Z-Score: {z:.2f} | Costo por Unidad: Q{costo_u:,.2f}")
            elif z > 1.5:
                st.warning(f"🟡 ALERTA MODERADA — Z-Score: {z:.2f} | Costo por Unidad: Q{costo_u:,.2f}")
            else:
                st.success(f"🟢 Sin Alerta — Z-Score: {z:.2f} | Costo por Unidad: Q{costo_u:,.2f}")

            # Metadata
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Departamento", depto)
            m2.metric("Sector", sector_s)
            m3.metric("Especie", acc.get('especie','N/D'))
            m4.metric("Etapa", acc.get('etapa_actual','N/D'))
            m5,m6,m7,m8 = st.columns(4)
            m5.metric("Monto Ejecutado", f"Q{acc['monto_ejecutado']:,.0f}")
            m6.metric("Meta Ejecutada", f"{acc['meta_ejecutada']:,.1f}")
            m7.metric("Ejec. Física", f"{acc['tasa_ejec_fisica']:.1%}" if pd.notna(acc.get('tasa_ejec_fisica')) else "N/D")
            m8.metric("Ejec. Presupuestal", f"{acc['tasa_ejec_presupuestal']:.1%}" if pd.notna(acc.get('tasa_ejec_presupuestal')) else "N/D")
            st.divider()

            # Peer comparison
            st.subheader("Comparativa con Proyectos Similares")
            df_peers = plot_total[
                (plot_total['departamento'] == depto) &
                (plot_total['sector'] == sector_s)
            ].copy()

            if len(df_peers) > 1:
                q1p = df_peers['cost_total_por_unidad'].quantile(0.25)
                q3p = df_peers['cost_total_por_unidad'].quantile(0.75)
                upper_fence = q3p + 1.5 * (q3p - q1p)
                peer_median = df_peers['cost_total_por_unidad'].median()
                df_peers['_color'] = df_peers['snip'].apply(
                    lambda s: '#ef4444' if s == snip_id else '#93c5fd')
                df_peers_s = df_peers.sort_values('cost_total_por_unidad')
                fig_p = go.Figure()
                fig_p.add_trace(go.Bar(
                    x=df_peers_s['cost_total_por_unidad'],
                    y=df_peers_s['snip'].astype(str),
                    orientation='h', marker_color=df_peers_s['_color'],
                    hovertemplate='SNIP: %{y}<br>Costo: Q%{x:,.2f}<extra></extra>'))
                fig_p.add_vline(x=peer_median, line_dash='dash', line_color='gray',
                    annotation_text=f'Mediana: Q{peer_median:,.0f}')
                fig_p.add_vline(x=upper_fence, line_dash='dot', line_color='#f59e0b',
                    annotation_text=f'Límite IQR: Q{upper_fence:,.0f}')
                fig_p.update_layout(template='plotly_white', showlegend=False,
                    title=f'Pares en {depto} / {sector_s}  (rojo = seleccionado)',
                    height=max(300, len(df_peers_s) * 28))
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("No hay suficientes proyectos similares.")

            # Provider
            st.divider()
            st.subheader("Proveedor Adjudicado")
            df_gc_s = df_gc[df_gc['snip'] == snip_id] if not df_gc.empty else pd.DataFrame()
            if not df_gc_s.empty and COL_PROVEEDOR in df_gc_s.columns:
                prov_n = df_gc_s[COL_PROVEEDOR].iloc[0]
                nog    = df_gc_s[COL_NOG].iloc[0] if COL_NOG in df_gc_s.columns else 'N/D'
                adj    = df_gc_s[COL_ADJUDICADO].iloc[0] if COL_ADJUDICADO in df_gc_s.columns else 0
                p1,p2,p3 = st.columns(3)
                p1.metric("Proveedor", prov_n)
                p2.metric("NOG", nog)
                p3.metric("Adjudicado (Q)", f"Q{adj:,.0f}")

                otros = df_gc[df_gc[COL_PROVEEDOR] == prov_n]['snip'].nunique()
                st.markdown(f"**{prov_n}** tiene **{otros}** proyecto(s) total en el dataset.")

                otros_df = (
                    df_gc[df_gc[COL_PROVEEDOR] == prov_n][['snip']]
                    .merge(plot_total[['snip', 'departamento','cost_total_por_unidad','z_score']], on='snip', how='inner')
                    #.query("snip != @snip_id")
                )
                if not otros_df.empty:
                    n_oa = (otros_df['z_score'] > 1.5).sum()
                    if n_oa > 0:
                        st.warning(f"⚠️ Este proveedor tiene {n_oa} proyecto(s) con alerta.")
                    with st.expander(f"Ver todos los proyectos de {prov_n}"):
                        st.dataframe(otros_df.sort_values('z_score', ascending=False).rename(columns = 
                            {'cost_total_por_unidad': 'Costo por Unidad (Q)'})
                            .style.apply(style_outliers, axis=1).format(precision=2, thousands=","),
                            use_container_width=True)

                # Concentration check
                if not df_prov.empty and COL_PROVEEDOR in df_prov.columns:
                    tot_d = df_prov[df_prov['departamento'] == depto]['monto_ejecutado'].sum()
                    prov_d = df_prov[(df_prov['departamento'] == depto) &
                        (df_prov[COL_PROVEEDOR] == prov_n)]['monto_ejecutado'].sum()
                    if tot_d > 0:
                        pct_c = prov_d / tot_d * 100
                        if pct_c > 50:
                            st.error(f"🔴 Concentración: {prov_n} controla el {pct_c:.1f}% del monto en {depto}.")
                        elif pct_c > 30:
                            st.warning(f"🟡 {prov_n} tiene el {pct_c:.1f}% del monto en {depto}.")
            else:
                st.info("Sin datos en Guatecompras para este proyecto.")