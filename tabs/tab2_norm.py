import streamlit as st
import stats_engine as engine
import scipy.stats as stats
import plotly.express as px

def render(df_raw, val_col):
    st.header("📊 2단계: 통계적 정규성(Normality Test) 적합 검증")
    st.write("공정능력지수 산출의 대전제인 정규분포 적합성을 다각도로 검증합니다.")
    
    p_value, is_normal = engine.run_normality_test(df_raw[val_col].values)
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("Shapiro-Wilk 정밀 검정 결과")
        st.metric(label="Shapiro-Wilk Test p-value", value=f"{p_value:.6f}")
        
        if is_normal:
            st.success("✅ 정규성 만족: 변환 없이 원본 데이터 상태로 분석을 안전하게 수행합니다.")
        else:
            st.warning("⚠️ 정규성 불만족 (p < 0.05): 공정능력지수 신뢰도를 확보하기 위해 다음 단계에서 Box-Cox 변환이 권장됩니다.")
            
    with col2:
        st.subheader("정규성 시각화 판별을 위한 Q-Q Plot")
        z_value = stats.zscore(df_raw[val_col].values)
        (x, y), _ = stats.probplot(z_value, dist='norm')
        
        fig_qq = px.scatter(
            x=x, y=y, title='Theoretical vs Sample Quantiles 직선 대조',
            labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}, template='seaborn'
        )
        fig_qq.add_shape(type='line', x0=-3, y0=-3, x1=3, y1=3, line=dict(color='red', width=2))
        fig_qq.update_layout(width=420, height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_qq, use_container_width=True)
