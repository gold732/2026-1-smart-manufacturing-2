import streamlit as st
import stats_engine as engine
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# app.py와 개수를 똑같이 맞춰 TypeError 소멸시킴
def render(df_raw, sg_col, val_col, lsl, usl, sigma_method):
    st.header("📊 2단계 대시보드: 통계적 정규성 검정 및 변환 가시화")
    
    p_value, is_normal = engine.run_normality_test(df_raw[val_col].values)
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("Shapiro-Wilk 정밀 적합 검정")
        st.metric(label="Shapiro-Wilk Test p-value", value=f"{p_value:.6f}")
        if is_normal:
            st.success("✅ 정규성 조건 만족 (p-value >= 0.05)")
        else:
            st.warning("⚠️ 정규성 조건 불만족 (p-value < 0.05)")
            st.info("💡 강의록 지침: 공정 지수의 신뢰도를 높이기 위해 우측에 자동으로 Box-Cox 거듭제곱 변환을 실행하여 대조합니다.")
            
    with col2:
        st.subheader("정규성 분위수 대조 (Q-Q Plot)")
        z_value = stats.zscore(df_raw[val_col].values)
        (x, y), _ = stats.probplot(z_value, dist='norm')
        fig_qq = px.scatter(x=x, y=y, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}, template='seaborn')
        fig_qq.add_shape(type='line', x0=-3, y0=-3, x1=3, y1=3, line=dict(color='red', width=2))
        fig_qq.update_layout(width=380, height=380)
        st.plotly_chart(fig_qq, use_container_width=True)

    if not is_normal and (df_raw[val_col] > 0).all():
        st.subheader("🔄 강의록 28p 규격: Box-Cox 비선형 변환 전후 분포 패턴 정밀 비교")
        transformed_vals, lsl_t, usl_t, lambda_val = engine.apply_box_cox(df_raw[val_col].values, lsl, usl)
        
        fig_compare = make_subplots(rows=1, cols=2, subplot_titles=('원본 비대칭 데이터 분포', f'Box-Cox 변환 후 정규화분포 (λ = {lambda_val:.2f})'))
        fig_compare.add_trace(go.Histogram(x=df_raw[val_col], name='원본', marker_color='indianred', opacity=0.7), row=1, col=1)
        fig_compare.add_trace(go.Histogram(x=transformed_vals, name='변환', marker_color='steelblue', opacity=0.7), row=1, col=2)
        fig_compare.update_layout(height=350, template='seaborn', showlegend=False)
        st.plotly_chart(fig_compare, use_container_width=True)
