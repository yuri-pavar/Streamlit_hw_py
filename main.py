

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


DATA_PATH = 'data/bank_data.csv'
IMAGE_PATH = 'data/bank2.png'


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def df_perc_by_group(df, col):
    tt = df.groupby([col, "TARGET_TXT"], as_index=False).agg({'SOCSTATUS_WORK_FL':'count'})
    tt['perc'] = 100*round(tt.SOCSTATUS_WORK_FL/tt.groupby(col)['SOCSTATUS_WORK_FL'].transform(sum), 3)
    return tt


def show_age_dist():
    fig = px.histogram(df, x="AGE", color="TARGET_TXT",
                       color_discrete_map={'отклика нет': '#FFDEAD', 'отклик': '#7f7053'},
                       title='Распределение возраста клиентов',
                       text_auto=True,
                       pattern_shape="TARGET_TXT")
    fig['layout']['yaxis']['title'] = 'Количество клиентов'
    fig['layout']['xaxis']['title'] = 'Возраст'
    fig.update_layout(barmode='stack')

    return fig


def show_pers_inc_dist():
    fig = px.histogram(df, x="PERSONAL_INCOME",
                       color="TARGET_TXT",
                       color_discrete_map={'отклика нет': '#FFDEAD', 'отклик': '#7f7053'},
                       title='Распределение персонального дохода')

    fig['layout']['yaxis']['title'] = 'Количество клиентов'
    fig['layout']['xaxis']['title'] = 'Доход'

    return fig


def show_plots_by_child():
    tt = df.groupby(['CHILD_TOTAL', 'TARGET'], as_index=False) \
        .agg({'PERSONAL_INCOME': 'mean', 'LOAN_NUM_TOTAL': 'mean'})
    tt.PERSONAL_INCOME = tt.PERSONAL_INCOME.round()
    tt.LOAN_NUM_TOTAL = tt.LOAN_NUM_TOTAL.round(2)

    fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Средний доход',
                                    'Среднее количество заемов'))

    fig.add_trace(
        go.Scatter(x=tt.loc[tt.TARGET == 1].CHILD_TOTAL,
                   y=tt.loc[tt.TARGET == 1].PERSONAL_INCOME,
                   line=dict(color='#7f7053', width=2),
                   name='отклик'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=tt.loc[tt.TARGET == 0].CHILD_TOTAL,
                   y=tt.loc[tt.TARGET == 0].PERSONAL_INCOME,
                   line=dict(color='#FFDEAD', width=2),
                   name='отклика нет'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=tt.loc[tt.TARGET == 1].CHILD_TOTAL,
                   y=tt.loc[tt.TARGET == 1].LOAN_NUM_TOTAL,
                   line=dict(color='#7f7053', width=2),
                   showlegend=False),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=tt.loc[tt.TARGET == 0].CHILD_TOTAL,
                   y=tt.loc[tt.TARGET == 0].LOAN_NUM_TOTAL,
                   line=dict(color='#FFDEAD', width=2),
                   showlegend=False),
        row=1, col=2
    )

    fig['layout']['xaxis']['title'] = 'Количество детей'
    fig['layout']['xaxis2']['title'] = 'Количество детей'

    fig.update_layout(height=350, width=900,
                      title_text="Зависимость финансовых показателей от количества детей",
                      title_xanchor='auto', title_xref='paper')

    return fig


def show_client_plots():
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Наличие работы',
                                        'Выход на пенсию', 'Пол', 'Количество детей'))

    tt0 = df_perc_by_group(df, 'SOCSTATUS_WORK_FL_TXT')
    trace0 = go.Bar(x=tt0.loc[tt0.TARGET_TXT == 'отклик'].SOCSTATUS_WORK_FL_TXT,
                    y=tt0.loc[tt0.TARGET_TXT == 'отклик'].perc,
                    marker=dict(color='#7f7053'),
                    texttemplate="%{y:.2f}",
                    textfont_size=9,
                    name='отклик')
    trace0_1 = go.Bar(x=tt0.loc[tt0.TARGET_TXT == 'отклика нет'].SOCSTATUS_WORK_FL_TXT,
                      y=tt0.loc[tt0.TARGET_TXT == 'отклика нет'].perc,
                      marker=dict(color='#FFDEAD'),
                      texttemplate="%{y:.2f}",
                      textfont_size=9,
                      name='отклика нет')

    tt1 = df_perc_by_group(df, 'SOCSTATUS_PENS_FL_TXT')
    trace1 = go.Bar(x=tt1.loc[tt1.TARGET_TXT == 'отклик'].SOCSTATUS_PENS_FL_TXT,
                    y=tt1.loc[tt1.TARGET_TXT == 'отклик'].perc,
                    marker=dict(color='#7f7053'),
                    texttemplate="%{y:.2f}",
                    textfont_size=9,
                    showlegend=False)
    trace1_1 = go.Bar(x=tt1.loc[tt1.TARGET_TXT == 'отклика нет'].SOCSTATUS_PENS_FL_TXT,
                      y=tt1.loc[tt1.TARGET_TXT == 'отклика нет'].perc,
                      marker=dict(color='#FFDEAD'),
                      texttemplate="%{y:.2f}",
                      textfont_size=9,
                      showlegend=False)

    tt2 = df_perc_by_group(df, 'GENDER_TXT')
    trace2 = go.Bar(x=tt2.loc[tt2.TARGET_TXT == 'отклик'].GENDER_TXT,
                    y=tt2.loc[tt2.TARGET_TXT == 'отклик'].perc,
                    marker=dict(color='#7f7053'),
                    texttemplate="%{y:.2f}",
                    textfont_size=9,
                    showlegend=False)
    trace2_1 = go.Bar(x=tt2.loc[tt2.TARGET_TXT == 'отклика нет'].GENDER_TXT,
                      y=tt2.loc[tt2.TARGET_TXT == 'отклика нет'].perc,
                      marker=dict(color='#FFDEAD'),
                      texttemplate="%{y:.2f}",
                      textfont_size=9,
                      showlegend=False)

    tt3 = df_perc_by_group(df, 'CHILD_TOTAL')
    trace3 = go.Bar(x=tt3.loc[tt3.TARGET_TXT == 'отклик'].CHILD_TOTAL,
                    y=tt3.loc[tt3.TARGET_TXT == 'отклик'].perc,
                    marker=dict(color='#7f7053'),
                    texttemplate="%{y:.2f}",
                    textfont_size=9,
                    showlegend=False)
    trace3_1 = go.Bar(x=tt3.loc[tt3.TARGET_TXT == 'отклика нет'].CHILD_TOTAL,
                      y=tt3.loc[tt3.TARGET_TXT == 'отклика нет'].perc,
                      marker=dict(color='#FFDEAD'),
                      texttemplate="%{y:.2f}",
                      textfont_size=9,
                      showlegend=False)

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace0_1, 1, 1)

    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace1_1, 1, 2)

    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace2_1, 2, 1)

    fig.append_trace(trace3, 2, 2)
    fig.append_trace(trace3_1, 2, 2)

    fig['layout']['yaxis']['title'] = '% клиентов'
    fig['layout']['yaxis2']['title'] = '% клиентов'
    fig['layout']['yaxis3']['title'] = '% клиентов'
    fig['layout']['yaxis4']['title'] = '% клиентов'

    fig.update_layout(height=500, width=1400, title_text="Распределения клиентских данных",
                      title_xanchor='auto', title_xref='paper')

    return fig


def show_plots_by_loans():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Процент клиентов от кол-ва заемов',
                                        'Процент клиентов без долгов от кол-ва заемов'))

    tt = df_perc_by_group(df, 'LOAN_NUM_TOTAL')

    trace0 = go.Bar(x=tt.loc[tt.TARGET_TXT == 'отклик'].LOAN_NUM_TOTAL,
                    y=tt.loc[tt.TARGET_TXT == 'отклик'].perc,
                          marker=dict(color='#7f7053'),
                          texttemplate="%{y:.2f}",
                          textfont_size=9,
                          name='отклик')
    trace0_1 = go.Bar(x=tt.loc[tt.TARGET_TXT == 'отклика нет'].LOAN_NUM_TOTAL,
                        y=tt.loc[tt.TARGET_TXT == 'отклика нет'].perc,
                            marker=dict(color='#FFDEAD'),
                            texttemplate="%{y:.2f}",
                            textfont_size=9,
                            name='отклика нет')

    tt = df[['TARGET', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']].copy()
    tt['is_all_closed'] = 1
    tt.loc[tt.LOAN_NUM_TOTAL != tt.LOAN_NUM_CLOSED, 'is_all_closed'] = 0
    tt2 = tt.groupby(['TARGET', 'LOAN_NUM_TOTAL'], as_index=False).agg({'is_all_closed': 'mean'})

    trace1 = go.Scatter(x=tt2.loc[tt2.TARGET == 1].LOAN_NUM_TOTAL,
                        y=tt2.loc[tt2.TARGET == 1].is_all_closed,
                        line=dict(color='#7f7053', width=2), showlegend=False)

    trace1_1 = go.Scatter(x=tt2.loc[tt2.TARGET == 0].LOAN_NUM_TOTAL,
                          y=tt2.loc[tt2.TARGET == 0].is_all_closed,
                          line=dict(color='#FFDEAD', width=2), showlegend=False)

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace0_1, 1, 1)

    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace1_1, 1, 2)

    fig['layout']['yaxis']['title'] = '% клиентов'
    fig['layout']['yaxis2']['title'] = '% клиентов без долгов'
    fig['layout']['xaxis']['title'] = 'Количество заемов'
    fig['layout']['xaxis2']['title'] = 'Количество заемов'

    fig.update_layout(height=450, width=1300, title_text="Зависимость финансовых показателей от количества заемов",
                      title_xanchor='auto', title_xref='paper')

    return fig


def show_plots_correlation():
    fig = px.imshow(df.drop('AGREEMENT_RK', axis=1).select_dtypes(include=np.number).corr().round(2),
                    text_auto=True,
                    color_continuous_scale=['#FFDEAD', '#7f7053', '#80461B']
                    )
    fig.layout.height = 800
    fig.layout.width = 800

    return fig

# заголовок
image = Image.open(IMAGE_PATH)
st.set_page_config(
        # layout="centered",
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Bank Promo",
        page_icon=image)
st.title('Исследование эффективности промо-кампании')
st.write('Цель исследования - определить сегмент клиентов, которые откликнулись на предложение, а также предложить модель, которая бы могла определять похожих клиентов')
st.image(image)

# загрузка данных
df = load_data()

# заголовок шторки
st.sidebar.title('Портрет клиента')

# фильтр на пол в шторке
with st.sidebar:
    radio_gender = st.radio("Выберите пол", ('Оба :family:', 'Мужской :boy:', 'Женский :girl:'))
if radio_gender == 'Мужской :boy:':
    df = df.loc[df['GENDER'] == 1]
elif radio_gender == 'Женский :girl:':
    df = df.loc[df['GENDER'] == 0]

# фильтр на возраст в шторке
min_age = min(df['AGE'])
max_age = max(df['AGE'])
selected_age_l, selected_age_r = st.sidebar.slider("Выберите возраст",
                                                   min_value=min_age, max_value=max_age,
                                                   value=(min_age, max_age))
df = df.loc[(df['AGE'] >= selected_age_l) & (df['AGE'] <= selected_age_r)]

# фильтр на доход в шторке
min_inc = min(df['PERSONAL_INCOME'])
max_inc = max(df['PERSONAL_INCOME'])
selected_income_l, selected_income_r = st.sidebar.slider("Выберите доход",
                                                         min_value=min_inc, max_value=max_inc,
                                                         value=(min_inc, max_inc))
df = df.loc[(df['PERSONAL_INCOME'] >= selected_income_l) & (df['PERSONAL_INCOME'] <= selected_income_r)]

# основа
st.write('Далее все рассуждения будут приводиться для общего случая, т е всей выборки, которая участвовала в эксперименте.  '
         '\nБыло бы полезно также посмотреть некие разрезы аудитории для более детальной аналитики.  '
         '\nДля этого в левой части экрана помещены фильтры, с помощью которых можно поближе рассмотреть отдельные сегменты. '
)

st.subheader('Клиентские данные')
st.write('Посмотрим на некоторые показатели клиентов.  '
         '\nВажно понять, какие клиенты откликаются на наше предложение с точки зрения клиента как человека.  ')

st.plotly_chart(
    show_age_dist(),
    use_container_width=True
)
st.write('Больше всего наших клиентов находятся в группе от 25 до 40 лет, далее с увеличением возраста начинается заметное падение количества клиентов. Эта же группа  '
         '\nявляется и наиболее активной с точки зрения откликов в наше предложение, что логично, т к старшее поколение менее склонно к таким активностям.'
)

st.plotly_chart(
    show_client_plots(),
    use_container_width=True
)
st.write('На предложение откликнулись в большей степени только клиенты работающие и не находящиеся на пенсии. В тоже же время видно, что небольшой процент неработающих  '
         '\nклиентов и клиентов-пенсионеров все же заинтересовались предложением. Хотя эти показатели скорее всего пересекаются, стоит подробнее посмотреть на сегмент  '
         '\nпенсионеров, т к хотя бы какая-то активность от них не ожидается (если у них не большой доход). Интересный момент, что женщины чуть лучше отреагировали на  '
         '\nпредложение, чем мужчины. К зависимость от количества детей стоит также присмотреться, т к  для клиентов с >3 детьми процент постепенно повышается, но возможно,  '
         '\nэтот эффект случайный, т к все-таки не так много семей с таким количеством детей.'
)

st.subheader('Финансовые данные')
st.write('Посмотрим на финансовые показатели клиентов.  '
         '\nХочется понять распределение доходов клиентов, а также их взаимодействие с заемами в различных разрезах. ')

st.plotly_chart(
    show_pers_inc_dist(),
    use_container_width=True
)
st.write('Из графика видно, что наибольшее количество клиентов имеет доход до 50.000 (около 99%). Есть некоторые клиенты с внушительными доходами, однако это скорее  '
         '\nисключение. Также видим, что есть пики в 10.000 и 15.000. С точки зрения откликов не видно как-то зависимости, стоит покопать здесь грубже.'
)

st.plotly_chart(
    show_plots_by_loans(),
    use_container_width=True
)
st.write('Здесь хотелось посмотреть, как распределились клиенты в зависимости от количества заемов. Для >6 заемов видим увеличение процентного соотношения, но здесь,  '
         '\nкак и ситуации с детьми, эффект можем наблюдать случайно, т к таких клиентов всего 0.12%. Также хотелось посмотреть зависимость клиентов без долгов по заемам  '
         '\nот количества заемов с разделением на откликнувшихся и нет. Кажется, что практически всегда откликнувшиеся имеют меньший процент закрытия долгов (здесь помним,  '
         '\nчто >6 заемов имеют всего 0.12% клиентов).'
)

st.plotly_chart(
    show_plots_by_child(),
    use_container_width=True
)
st.write('На первом графике видим интересную зависимость - средний доход клиентов падает в зависимости от количества детей. Также интересно, что у отликнувшихся в среднем более  '
         '\nвысокий доход. Среднее количество заемов, кажется, особо не отличается в группах. Здесь для обоих случаев уточним, что  >5 детей имеют только 0.14% клиентов и их  '
         '\nрезультаты могут искажать реальную картину.'
)


st.subheader('Корреляция')
st.write('Из матрицы корреляций видим, что корреляции с целевой переменной околонулевые для всех признаков. Наиболее выделяются здесь отрицательная зависимость от возраста и  '
         '\nпрактически идентичная положительная с доходом. '
         '\n  '
         '\nНаиболее высокие корреляции признаков следующие:  '
         '\n  - наличие работы - пенсия  '
        '\n  - возраст - наличие работы  '
        '\n  - возраст - пенсия  '
        '\n  - количество детей - количество иждивенцев  '
        '\n  - количество заемов - количество закрытых заемов  '
)
st.plotly_chart(
    show_plots_correlation(),
    use_container_width=True
)

