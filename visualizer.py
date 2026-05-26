import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

def plot_boxplot_dashboard(df, x_col, y_col):
    """데이터 분포 조감을 위한 Boxplot 생성"""
    fig = px.box(
        df, x=x_col, y=y_col, color=x_col, points='all',
        title=f'[{y_col}] 부품/라인별 박스플롯 분포 분석',
        template='seaborn'
    )
    fig.update_layout(width=800, height=450)
    return fig

def plot_qq_dashboard(values):
    """정규성 판별 가시화를 위한 Q-Q Plot 시각화"""
    z_value = stats.zscore(values)
    (x, y), _ = stats.probplot(z_value, dist='norm')
    
    fig = px.scatter(
        x=x, y=y, title='정규성 검증을 위한 Q-Q Plot',
        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
        template='seaborn'
    )
    fig.add_shape(type='line', x0=-3, y0=-3, x1=3, y1=3, line=dict(color='red', width=2))
    fig.update_layout(width=400, height=400)
    return fig

def plot_process_capability_histogram(df, val_col, lsl, usl, metrics):
    """LSL/USL선 가이드 및 정량 평가지표 수치 테이블 박스가 동봉된 히스토그램 시각화"""
    fig = px.histogram(
        df, x=val_col, nbins=20, marginal="box",
        title=f'{val_col.capitalize()} 공정능력분석 히스토그램 종합 리포트',
        opacity=0.6, template='seaborn'
    )
    fig.add_vline(x=lsl, line_width=2, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=usl, line_width=2, line_dash="dash", line_color="red", annotation_text="USL")
    
    # 스펙 스코어 박스 텍스트 포맷 구성
    annotation_text = (
        f"<b>공정 지표 스코어</b><br>"
        f"Cp = {metrics['Cp']:.4f}<br>"
        f"Cpk = {metrics['Cpk']:.4f}<br>"
        f"Pp = {metrics['Pp']:.4f}<br>"
        f"Ppk = {metrics['Ppk']:.4f}"
    )
    fig.add_annotation(
        xref='paper', yref='paper', x=0.98, y=0.85,
        text=annotation_text, showarrow=False, align="left",
        bgcolor="white", bordercolor="black", borderwidth=1
    )
    fig.update_layout(width=800, height=500)
    return fig

def plot_control_chart_dashboard(chart1, chart2, chart_type='Xbar-R'):
    """통계적 관리한계(UCL/CL/LCL) 점선 추적 및 이탈 데이터 자동 식별 모니터링 관리도"""
    names = chart_type.split('-')
    sub_title1 = names[0]
    sub_title2 = names[1] if len(names) > 1 else 'MR'
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'{sub_title1} 관리도', f'{sub_title2} 관리도'))
    
    # 상단 메인 데이터 차트
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['point'], mode='lines+markers', name=sub_title1, marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=1, col=1)
    
    # 하단 산포/변동 범위 차트
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['point'], mode='lines+markers', name=sub_title2, marker=dict(size=6, color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=2, col=1)
    
    fig.update_layout(width=850, height=650, title_text=f"{chart_type} 통계적 모니터링 관리 시각화 대시보드", showlegend=False, template='seaborn')
    return fig