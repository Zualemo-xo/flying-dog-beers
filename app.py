'''import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

########### Define your variables
beers=['Chesapeake Stout', 'Snake Dog IPA', 'Imperial Porter', 'Double Dog IPA']
ibu_values=[35, 60, 85, 75]
abv_values=[5.4, 7.1, 9.2, 4.3]
color1='lightblue'
color2='darkgreen'
mytitle='Beer Comparison'
tabtitle='beer!'
myheading='Flying Dog Beers'
label1='IBU'
label2='ABV'
githublink='https://github.com/austinlasseter/flying-dog-beers'
sourceurl='https://www.flyingdog.com/beers/'

########### Set up the chart
bitterness = go.Bar(
    x=beers,
    y=ibu_values,
    name=label1,
    marker={'color':color1}
)
alcohol = go.Bar(
    x=beers,
    y=abv_values,
    name=label2,
    marker={'color':color2}
)

beer_data = [bitterness, alcohol]
beer_layout = go.Layout(
    barmode='group',
    title = mytitle
)

beer_fig = go.Figure(data=beer_data, layout=beer_layout)


########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading),
    dcc.Graph(
        id='flyingdog',
        figure=beer_fig
    ),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A('Data Source', href=sourceurl),
    ]
)

if __name__ == '__main__':
    app.run_server()'''

#vid 5
#working code for live graph vidp4
'''import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)


app = dash.Dash(__name__)
server = app.server
app.title='hello'
app.layout = html.Div(
    [
        html.H1("Live Sentiment Graph",style={"background-color":"red"}),
        dcc.Input(id='sentiment_term', value='lockdown', type='text'),
        dcc.Graph(id='live-graph', animate=False),
        dcc.Interval(
            id='graph-update',
            interval=1000,
            n_intervals = 0
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
              [
              Input(component_id='sentiment_term', component_property='value'),
              Input('graph-update', 'n_intervals')]
              )
def update_graph_scatter(sentiment_term,n):
    try:
        conn = sqlite3.connect('assets/twitter.db',check_same_thread=False)
        #c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + sentiment_term + '%',))
        df.sort_values('unix', inplace=True)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/2)).mean()

        df['date'] = pd.to_datetime(df['unix'],unit='ms')
        df.set_index('date', inplace=True)

        #df = df.resample('1min').mean()
        df.dropna(inplace=True)
        X = df.index
        Y = df.sentiment_smoothed

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )

        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                    yaxis=dict(range=[min(Y),max(Y)]),
                                                    title='Term: {}'.format(sentiment_term))}

    except Exception as e:
        print(str(e))
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

if __name__ == '__main__':
    app.run_server()'''

'''#vid4
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)


app = dash.Dash(__name__)
server = app.server
app.title='hello'
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1000,
            n_intervals = 0
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals')])


def update_graph_scatter(n):
    X.append(X[-1]+1)
    Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))

    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )

    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[min(Y),max(Y)]),)}





if __name__ == '__main__':
    app.run_server()'''



import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go #uc


#from credentials import plotly_token  
plotly_token='pk.eyJ1IjoiamlsbC1hbXVkaGluaSIsImEiOiJja2JxdzdxbTcwMWYxMnNxeDgwNTFyaDY0In0.EEQoKyRRzHxKypU4jSGBHA'

import dash
import dash.dependencies as dd
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np

#!pip install wordcloud
from io import BytesIO
from PIL import Image
from wordcloud import WordCloud
import base64 #uc

import nltk
from nltk.probability import FreqDist

nltk.download('stopwords')
nltk.download('words')

df=pd.read_csv('etweets.csv')

app = dash.Dash(__name__)
server = app.server
app.title='hello'
app.layout = html.Div(style={'margin-right': '3%','margin-left':'3%','margin-top':'2%','margin-bottom':'2%'},
        children=[
    html.H2(children='Sentiment Analysis of COVID-19 Tweets',
            style={'text-align':'center'},className='bg-primary rounded-top border border-primary align-middle'),
    
    dbc.Card([
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Home", tab_id="tab-1"),
                    dbc.Tab(label="Search by State", tab_id="tab-2"),
                    dbc.Tab(label="Search by Keyword", tab_id="tab-3"),
                    dbc.Tab(label="Headlines", tab_id="tab-5"),
                    dbc.Tab(label="About", tab_id="tab-4"),
                ],
                id="card-tabs",
                card=True,
                active_tab="tab-1"
            )
        ),
        dbc.CardBody(html.P(id="card-content"),style={'background':'black'},className="border border-secondary"),
    ])])


def update_file():
    global df
    df=pd.read_csv('etweets.csv')

@app.callback(dd.Output("card-content", "children"), [dd.Input("card-tabs", "active_tab")])
def tab_content(active_tab):
    if active_tab=='tab-1':
        home= [
        html.Div([dcc.Graph(id='live-update-graph'),
                 
                  dcc.Interval(id='interval-component',interval=5*1000, n_intervals=0)],className='rounded'),
        html.Div([
        html.Div(children=[dcc.Graph(id='tweets-world')],
             style={'width': '40%'},className='h-50 d-inline-block rounded border border-white align-self-center'),

        html.Div([dcc.Dropdown(id='tweet-choice',
                               options=[{'label': 'World', 'value': 'world'},
                                        {'label': 'India', 'value': 'IN'}], value='IN'),

              dcc.Graph(id='tweets-map')],
             style={'width': '50%'},className='h-50 d-inline-block rounded align-middle border border-white'),
        ],style={'margin-top':'2%','margin-bottom':'2%'},
            className='d-flex justify-content-around align-self-center '),
        html.Div([
            html.Div(children=[dcc.Graph(id='sentiments-pie-chart')],
             style={'width': '35%'},className='h-50 d-inline-block rounded border border-white'),
            html.Div(children=[html.Img(id="overallsentiment_wc")],
                     style={'width': '60%'},className='h-50 d-inline-block rounded align-middle border border-white')
        ],style={'margin-top':'2%','margin-bottom':'2%'},
            className='d-flex justify-content-around align-self-center ')]
        
        return home
    
    
    elif active_tab=='tab-2':
        states=[                                                                                             
            html.Div(dcc.Graph(id='Figure1',style={'display':'block','width':'90%','height':'60%', 
                                                             'margin-left':'auto','margin-right': 'auto'})),
                                                                                                                       
            html.Div(dbc.Alert('Analysis of a state',color='blue'),className="m-1",style={'display':'block','color':'red','fontSize':30, 'text-align': 'center','font-family': "Tmes New roman"}),
                                                                                                                       
            html.Div(dcc.Dropdown(id='group-select',value='Telangana'),
                     style={"display":'block',"width":'30%',"color":"black","font-size":"20px","border-style":"groove","float":"center",
                         'text-align': 'center','margin-left':'auto','margin-right': 'auto','vertical-align':'middle'}),
                                                                                                                       
            html.Div(dcc.Graph(id="graph",style={"display":"block","width":"80%","margin-top":"2%" ,'margin-left':'auto','margin-right': 'auto','vertical-align':'middle'})),
                                                                                                                       
            html.Div(dcc.Graph(id="graph1"),style={"width":"50%","margin-top":"2%" ,'margin-left':'auto','margin-right': 'auto','vertical-align':'middle'})]
    
        
        return states
        
    elif active_tab=='tab-3':
        keyword=[
            html.Div([dbc.FormGroup([
                    dbc.Label("Enter the keyword",style={"color":"white","font-size":"200%","width":"20%"}),
                    dbc.Input(id="word_input",value="corona",placeholder="Text something here..", type="text",
                  style={"color":"black","font-size":"20px","border-style":"groove","display":"block","width":"19%"}),]),
                      dcc.Graph(id="Graph",
                    style={"paper_bgcolor":"black","display":"inline-block","width":"70%","float":"middle","height":"30%"})]),
                      
                      
            html.Div(dcc.Graph(id='polar_bar_graph',
                               style={'fontColor':'#DDDDDD','backgroundColor':'#111111','display':'block',
                                      'width':'70%', 'margin-left':'auto','margin-right': 'auto'})),
            html.Div([
                    html.Img(id="positive_wc",style={'width': '25%'},className="h-25 d-inline-block border border-secondary rounded-circle"),
                    html.Img(id="neutral_wc",style={'width': '25%'},className="h-25 d-inline-block border border-secondary rounded-circle"),
                    html.Img(id="negative_wc",style={'width': '25%'},className="h-25 d-inline-block border border-secondary rounded-circle")
                    ],id="wordcloud",className='d-flex justify-content-around'),
            
            ]
        
        return keyword
        

'''
#------------------------------------ HOME ------------------------------------

@app.callback(dash.dependencies.Output('live-update-graph', 'figure'),[dash.dependencies.Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    update_file()
    figure=px.line(df[-100:],x="Created_at",y="Sentiment_compound",template='plotly_dark')
    return figure
    
 
@app.callback(dd.Output('tweets-map', 'figure'), [dd.Input('tweet-choice', 'value')])
def return_fig_map(map_choice):
    if map_choice=='IN':
        df1=df[df['Country_code']=='IN']
        z=3
    else:
        df1=df
        z=1
    fig = px.scatter_mapbox(df1, lat="Latitude", lon="Longitude", color="Overall_sentiment",
                            hover_data=["Sentiment_compound", "Created_at"],zoom=z,template='plotly_dark',
                            color_discrete_map={'Positive':'rgb(128, 255, 0)',
                                                'Negative':'rgb(255, 51, 51)',
                                                'Neutral':'rgb(174, 179, 179)'})

    fig.update_layout(mapbox_style="dark", mapbox_accesstoken=plotly_token)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(dd.Output('tweets-world', 'figure'), [dd.Input('tweets-world', 'id')])
def return_fig_world(id):
    data=[go.Scattergeo(
            lat=df[df['Overall_sentiment']=='Positive']['Latitude'],
            lon=df[df['Overall_sentiment']=='Positive']['Longitude'],
            mode='markers',
            marker_color='rgb(128, 255, 0)',
            marker_size=5,
            name='Positive'),
     go.Scattergeo(
            lat=df[df['Overall_sentiment']=='Negative']['Latitude'],
            lon=df[df['Overall_sentiment']=='Negative']['Longitude'],
            mode='markers',
            marker_color='rgb(255, 51, 51)',
            marker_size=5,
            name='Negative'),
    go.Scattergeo(
            lat=df[df['Overall_sentiment']=='Neutral']['Latitude'],
            lon=df[df['Overall_sentiment']=='Neutral']['Longitude'],
            mode='markers',
            marker_color='rgb(174, 179, 179)',
            marker_size=5,
            name='Neutral')]

    layout =go.Layout(template='plotly_dark',
        geo=go.layout.Geo(
            showland = True,
            showcountries = True,
            showocean = True,
            countrywidth = 0.5,
            landcolor = 'rgb(51, 102, 0)',
            oceancolor = 'rgb(0, 0, 102)',
            projection_type='orthographic',
            center_lon=50,
            center_lat=0,
            projection_rotation_lon=50
        ))
    lon_range = np.arange(-180, 180, 2)

    frames = [go.Frame(layout=go.Layout(geo_center_lon=lon,geo_projection_rotation_lon =lon,geo_center_lat=0,geo_projection_rotation_lat=0),
                   name =f'{k+1}') for k, lon in enumerate(lon_range)]


    sliders = [dict(steps = [dict(method= 'animate',args= [[f'{k+1}'], 
                                          dict(mode= 'immediate',
                                          frame= dict(duration=0, redraw= True),
                                          transition=dict(duration= 0))],label=f'{k+1}') for k in range(len(lon_range))], 
                transition= dict(duration= 0 ),
                x=0, # slider starting position  
                y=0,   
                len=1.0) #slider length
              ]

    fig = go.Figure(data=data, layout=layout, frames=frames)
    fig.update_layout(sliders=sliders)
    return fig

@app.callback(dd.Output('sentiments-pie-chart', 'figure'), [dd.Input('sentiments-pie-chart', 'id')])
def return_pie_chart(id):
    df3=df['Overall_sentiment'].value_counts()
    df3=df3.rename_axis('index').reset_index()
    fig = go.Figure(data=[go.Pie(labels=['Positive','Negative','Neutral'], values=df3.Overall_sentiment, hole=.6)])
    fig.update_traces(hoverinfo='label+percent', textinfo='label', textfont_size=20,
                  marker=dict(colors=['rgb(34, 139, 34)', 'rgb(255, 69, 0)', 'rgb(65, 105, 225)'],
                              line=dict(color='rgb(245, 255, 250)', width=2)))
    fig.update_layout(template='plotly_dark',height=400)
    return fig


@app.callback(dd.Output('overallsentiment_wc', 'src'),[dd.Input('overallsentiment_wc', 'id')])
def return_image(id):
    img = BytesIO()
    fd =return_tweet_words(df)
    d= pd.DataFrame({'words': list(fd.keys()),'Count' : list(fd.values())})
    d= d.nlargest(columns = 'Count', n = 70)
    mask1=np.array(Image.open("mask1.png"))
    plot_wordcloud(data=d,mask=mask1,contour_color='white').save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())



#------------------------------------ SEARCH BY STATE ------------------------------------

@app.callback(dd.Output("Figure1",'figure'),[dd.Input("Figure1",'id')])
def graph(id):
    df1=df[df['Country_code']=='IN']
    
    a=pd.DataFrame()
    a['State']=df1['State'].unique()
    a['total_count']=a['State'].apply(lambda x: len(df1[df1['State']==x]))
    a['positive_count']=a['State'].apply(lambda x: len(df1[df1['State']==x]['State'][df1[df1['State']==x]['Overall_sentiment']=='Positive']))
    a['negative_count']=a['State'].apply(lambda x: len(df1[df1['State']==x]['State'][df1[df1['State']==x]['Overall_sentiment']=='Negative']))
    a['neutral_count']=a['State'].apply(lambda x: len(df1[df1['State']==x]['State'][df1[df1['State']==x]['Overall_sentiment']=='Neutral']))
    a.sort_values(by=['total_count'], inplace=True,ascending=False)

    trace3 = go.Bar(x =a.State.head(10), y = a.positive_count.head(10), name='positive')
    trace2 = go.Bar(x= a.State.head(10), y = a.negative_count.head(10), name ='negative')
    trace1 = go.Bar(x = a.State.head(10), y = a.neutral_count.head(10), name ='neutral' )
    data = [trace1, trace2, trace3]
    layout = go.Layout(barmode = 'group',paper_bgcolor="Black")
    fig= go.Figure(data = data, layout = layout)
    return(fig)

@app.callback(dd.Output("group-select",'options'),[dd.Input("group-select",'id')])
def preprocessing(id):
    df1=df[df['Country_code']=='IN']
    p=[];
    p=df1['State'].unique()
    options_list=[];
    for j in p:
        options_list.append({'label':j,'value':j})
    #options_list=options_list[0:6]
    return(options_list)


@app.callback([dd.Output("graph",'figure'),dd.Output("graph1",'figure')],[dd.Input('group-select','value')])

def update_output_div(m):
    df1=df[df['State']==m]
    
    f=return_tweet_words(df1)
    sort_orders = sorted(f.items(), key=lambda x: x[1], reverse=True)
    x=[];y=[]
    for j in range(0,10):
        x.append(sort_orders[j][0]);
        y.append(sort_orders[j][1]);
        
    y1=[];y2=[];y3=[];x1=[]
    
    for k in range(0,7):
        p=0;n=0;l=0;
        for j in range(0,df.shape[0]):
            if(df['State'][j]==m and df['Overall_sentiment'][j]=='Positive' and df['Tweet_text'][j].count(x[k])>0):
                p+=df['Tweet_text'][j].count(x[k])
            elif(df['State'][j]==m and df['Overall_sentiment'][j]=='Negative'and df['Tweet_text'][j].count(x[k])>0):
                n+=df['Tweet_text'][j].count(x[k])
            elif(df['State'][j]==m and df['Overall_sentiment'][j]=='Neutral'and df['Tweet_text'][j].count(x[k])>0):
                l+=df['Tweet_text'][j].count(x[k])      
        if(p>0 or n>0 or l>0):
            y1.append(p);
            y2.append(n);
            y3.append(l);
            x1.append(x[k]);

    fig = go.Figure(go.Bar(x=x1, y=y3, name='Neutral'))
    fig.add_trace(go.Bar(x=x1, y=y2, name='Negative'))
    fig.add_trace(go.Bar(x=x1, y=y1, name='Positive'))
    fig.update_layout(barmode='stack')
    fig.update_layout(title="State Analysis", xaxis_title="Frequent Words Used", yaxis_title="Count",
                      font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
                       paper_bgcolor="Black")
   

    
    p=len(df1[df1['Overall_sentiment']=='Positive'])
    n=len(df1[df1['Overall_sentiment']=='Negative'])
    l=len(df1[df1['Overall_sentiment']=='Neutral'])
    
    
    fig1 = go.Figure(data=go.Scatterpolar(r=[p,n,l],theta=['positive','Negative','Neutral'],fill='toself'))
    fig1.update_layout(paper_bgcolor="Black",
                       polar=dict(bgcolor='#1e2130',radialaxis=dict(visible=True,)),
                       showlegend=False)

    return fig,fig1


#------------------------------------ SEARCH BY KEYWORD ------------------------------------

def return_tweet_words(df):
    words = set(nltk.corpus.words.words())
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words+=['https','http','www','www.','like','still','every','get','made','this','can','they','amp','yet','may','also']

    tweet_words=[]
    for line in list(df["Tweet_text"]):
        for word in line.split():
            tweet_words.append(word.lower())
    
    tw=[w for w in tweet_words if not w in stop_words and w in words and len(w)>2 and not(w.isnumeric())]
    fdist=FreqDist(tw)
    
    return fdist


def plot_wordcloud(data,mask,contour_color):
    d = {a: x for a, x in data.values}
    wc = WordCloud(mask=mask,contour_width=3,contour_color=contour_color,background_color='black', width=480, height=360)
    wc.fit_words(d)
    return wc.to_image()


@app.callback([dd.Output('positive_wc', 'src'),dd.Output('neutral_wc', 'src'),dd.Output('negative_wc', 'src')], 
              [dd.Input('wordcloud', 'id')])
def make_image(b):
    df1=df[df["Country_code"]=='IN']
    
    img1 = BytesIO()
    df_p=df1[df1['Overall_sentiment'] =='Positive']
    fd_p =return_tweet_words(df_p)
    dp= pd.DataFrame({'words': list(fd_p.keys()),'Count' : list(fd_p.values())})
    dp= dp.nlargest(columns = 'Count', n = 40)
    mask1=np.array(Image.open("happy_wc.png"))
    plot_wordcloud(data=dp,mask=mask1,contour_color='green').save(img1, format='PNG')
    happy='data:image/png;base64,{}'.format(base64.b64encode(img1.getvalue()).decode())
    
    img2=BytesIO()
    df_n=df1[df1['Overall_sentiment'] =='Negative']
    fd_n =return_tweet_words(df_n)
    dn= pd.DataFrame({'words': list(fd_n.keys()),'Count' : list(fd_n.values())})
    dn= dn.nlargest(columns = 'Count', n = 40)
    mask2=np.array(Image.open("sad_wc.png"))
    plot_wordcloud(data=dn,mask=mask2,contour_color='red').save(img2, format='PNG')
    sad='data:image/png;base64,{}'.format(base64.b64encode(img2.getvalue()).decode())
    
    img3=BytesIO()
    df_ne=df1[df1['Overall_sentiment'] =='Neutral']
    fd_ne =return_tweet_words(df_ne)
    dne= pd.DataFrame({'words': list(fd_ne.keys()),'Count' : list(fd_ne.values())})
    dne= dne.nlargest(columns = 'Count', n = 40)
    mask3=np.array(Image.open("neutral_wc.png"))
    plot_wordcloud(data=dne,mask=mask3,contour_color='white').save(img3, format='PNG')
    neutral='data:image/png;base64,{}'.format(base64.b64encode(img3.getvalue()).decode())
    
    return happy,neutral,sad


@app.callback(dash.dependencies.Output("Graph",'figure'),[dash.dependencies.Input('word_input','value')])
def update_word(h):
    def search(t):
        d1=t.count(h.lower())
        return d1
    
    df1=df[df["Country_code"]=='IN']
    df1["count"]=df1["Tweet_text"].apply(search)
    df2=df1[df1['count']>0]
    df3=df2.drop(['User_Id','Created_at','Country_code','Latitude','Longitude','Tweet_text','Sentiment_compound','Overall_sentiment'],axis=1)
    df3=df3.groupby('State').sum()
    df3.sort_values(by=['count'], inplace=True,ascending=False)
    df3=df3.rename_axis('index').reset_index()
    df3.rename(columns={"index":"state"})
    
    fig5=px.bar(df3.head(10),x='index',y='count',hover_data=['index','count'],
                color='count',color_continuous_scale=px.colors.sequential.Plasma_r,
                labels={'pop':'count'},height=400)
    fig5.update_layout(title="Keyword Analysis",
        xaxis_title="Top States",yaxis_title="Count",
        paper_bgcolor="black",font=dict(family="Courier New, monospace",size=20,color="#DDDDDD"))
    
    return fig5

@app.callback(dd.Output('polar_bar_graph', 'figure'), [dd.Input('polar_bar_graph', 'id')])
def return_pie_chart(id):
    fd =return_tweet_words(df)
    d = pd.DataFrame({'Hashtag': list(fd.keys()),
                  'Count' : list(fd.values())})
    d = d.nlargest(columns = 'Count', n = 10)
    fig= px.bar_polar(d, r='Count', theta='Hashtag',
                   color='Hashtag', template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)
    return fig'''




app.config['suppress_callback_exceptions'] = True

if __name__ == '__main__':
    app.run_server()
