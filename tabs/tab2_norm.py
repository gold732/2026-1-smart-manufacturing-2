import streamlit as st
import stats_engine as engine
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render(df_raw, sg_col, val_col):
    st.header("📊 통계적 정규성 검정 및 데이터 변환")
    st.write("공정능력지수 산출의 대전제인 데이터의 정규분포 적합성을 다각도로 검증합니다.")
    
    lsl = st.session_state.get('lsl', 0)
    usl = st.session_state.get('usl', 0)
    p_value, is_normal = engine.run_normality_test(df_raw[val_col].values)
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("Shapiro-Wilk 적합 검정")
        st.metric(label="Shapiro-Wilk Test p-value", value=f"{p_value:.6f}")
        if is_normal:
            st.success("✅ 정규성 만족: 원본 데이터 상태로 분석을 진행합니다.")
        else:
            st.warning("⚠️ 정규성 불만족: 공정 지수의 신뢰도를 확보하기 위해 Box-Cox 데이터 변환을 적용합니다.")
            
    with col2:
        st.subheader("정규성 분위수 대조 (Q-Q Plot)")
        z_value = stats.zscore(df_raw[val_col].values)
        (x, y), _ = stats.probplot(z_value, dist='norm')
        fig_qq = px.scatter(x=x, y=y, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}, template='seaborn')
        fig_qq.add_shape(type='line', x0=-3, y0=-3, x1=3, y1=3, line=dict(color='red', width=2))
        fig_qq.update_layout(width=380, height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_qq, use_container_width=True)

    if not is_normal and (df_raw[val_col] > 0).all():
        st.subheader("🔄 데이터 변환 전후 분포 패턴 정밀 비교")
        transformed_vals, lsl_t, usl_t, lambda_val = engine.apply_box_cox(df_raw[val_col].values, lsl, usl)
        
        fig_compare = make_subplots(rows=1, cols=2, subplot_titles=('원본 비대칭 데이터 분포', f'Box-Cox 변환 후 정규화 분포 (λ = {lambda_val:.2f})'))
        fig_compare.add_trace(go.Histogram(x=df_raw[val_col], name='원본', marker_color='indianred', opacity=0.7), row=1, col=1)
        fig_compare.add_trace(go.Histogram(x=transformed_vals, name='변환', marker_color='steelblue', opacity=0.7), row=1, col=2)
        fig_compare.update_layout(height=320, template='seaborn', showlegend=False)
        st.plotly_chart(fig_compare, use_container_width=True)
