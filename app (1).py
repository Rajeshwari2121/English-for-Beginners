import random
import re
from datetime import date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title='English for Begineers | English–Telugu', page_icon='📚', layout='wide')
BASE = Path(__file__).resolve().parent

st.markdown('''<style>
.stApp{background:linear-gradient(125deg,#faf5ff,#eff6ff 52%,#fff1f2)}
.block-container{padding-top:1.3rem;max-width:1250px}
h1,h2,h3{color:#5925a5!important}
.stButton>button{border-radius:14px;background:linear-gradient(100deg,#6d28d9,#db2777);color:white;border:0;font-weight:700;min-height:44px}
.stButton>button:hover{color:white;transform:translateY(-2px);box-shadow:0 8px 20px #7c3aed55}
[data-testid="stMetric"]{background:white;border-radius:16px;padding:16px;box-shadow:0 4px 18px #6d28d91c}
.hero{background:linear-gradient(110deg,#5b21b6,#9333ea,#db2777);padding:28px;border-radius:24px;text-align:center;color:white;animation:enter .8s ease}
.hero h1{color:white!important;font-size:clamp(30px,4vw,52px)}
.hero p{color:#fdf4ff;font-size:20px}
@keyframes enter{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.sentence{font-size:clamp(23px,2.7vw,34px);font-weight:750;line-height:1.8;padding:24px;border-radius:20px;margin:14px 0;animation:enter .7s ease;overflow-wrap:anywhere}
.en{background:linear-gradient(115deg,#ffedd5,#fef3c7);color:#9a3412}
.te{background:linear-gradient(115deg,#d1fae5,#ccfbf1);color:#065f46}
.thought{background:linear-gradient(110deg,#6d28d9,#9333ea,#db2777);border-radius:22px;padding:30px;text-align:center;color:white;animation:enter 1s ease}
.thought .quote{font-size:clamp(23px,3vw,34px);font-weight:800;color:white;margin:16px 0}
.thought .telugu{font-size:25px;color:#e4ffe9;line-height:1.8}
.thought .message{background:#ffffff30;border-radius:12px;padding:12px;margin-top:15px;color:white}
</style>''', unsafe_allow_html=True)


def clean(value):
    return escape(str(value)) if pd.notna(value) else ''


def pos_explanation(pos):
    return {'NOUN':'Names a person, place, thing or idea','PROPN':'Proper name','VERB':'Action or event','AUX':'Helping verb','ADJ':'Describes a noun','ADV':'Modifies a verb, adjective or adverb','DET':'Determiner','PRON':'Pronoun','ADP':'Preposition','CCONJ':'Coordinating conjunction','SCONJ':'Subordinating conjunction','PUNCT':'Punctuation','NUM':'Number'}.get(pos,'Other category')


@st.cache_data
def read_csv(path):
    data = pd.read_csv(path, encoding='utf-8-sig').fillna('')
    data.columns = data.columns.str.strip()
    required = ['Word','Meaning_English','Meaning_Telugu','Own_Sentence','Own_Sentence_Telugu']
    absent = [c for c in required if c not in data.columns]
    if absent:
        raise ValueError('Missing columns: '+', '.join(absent))
    for c in ['Part_of_Speech','Synonyms','Antonyms','Verb_Base_Form','Noun_Form','Adjective_Form','Adverb_Form']:
        if c not in data.columns:
            data[c] = ''
    for c in required+['Part_of_Speech','Synonyms','Antonyms','Verb_Base_Form','Noun_Form','Adjective_Form','Adverb_Form']:
        data[c] = data[c].astype(str).str.strip()
    data = data[data.Word.ne('')].reset_index(drop=True)
    data['Initial'] = data.Word.str[:1].str.upper()
    data['Word_Length'] = data.Word.str.len()
    data['Sentence_Words'] = data.Own_Sentence.str.split().str.len()
    return data


@st.cache_resource
def get_spacy():
    try:
        import spacy
        return spacy.load('en_core_web_sm'), None
    except (ImportError, OSError) as error:
        return None, str(error)


THOUGHTS = [
    ('Every expert was once a beginner.','ప్రతి నిపుణుడు ఒకప్పుడు ప్రారంభకుడే.','Start small, but never stop learning.'),
    ('Your effort today creates your success tomorrow.','ఈ రోజు నీ కృషి రేపటి విజయాన్ని సృష్టిస్తుంది.','Keep working toward your goals.'),
    ('Mistakes are proof that you are trying.','తప్పులు చేస్తున్నావంటే నువ్వు ప్రయత్నిస్తున్నావని అర్థం.','Learn from every mistake.'),
    ('Small progress is still progress.','చిన్న పురోగతి కూడా పురోగతే.','Celebrate every step forward.'),
    ('Believe in yourself and keep moving forward.','నిన్ను నువ్వు నమ్ముకుని ముందుకు సాగు.','Confidence grows through practice.'),
    ('Knowledge grows when you share it.','జ్ఞానాన్ని పంచుకున్నప్పుడు అది పెరుగుతుంది.','Learn something and teach someone.'),
    ('Consistency turns dreams into achievements.','నిరంతర కృషి కలలను విజయాలుగా మారుస్తుంది.','Practice a little every day.'),
    ('A new day is a new opportunity to learn.','ప్రతి కొత్త రోజు నేర్చుకోవడానికి ఒక కొత్త అవకాశం.','Make today meaningful.'),
    ('Patience and practice build confidence.','సహనం మరియు సాధన ఆత్మవిశ్వాసాన్ని పెంచుతాయి.','Keep practicing.'),
    ('Every question is a chance to understand more.','ప్రతి ప్రశ్న మరింత అర్థం చేసుకునే అవకాశం.','Stay curious.')
]


def thought_card():
    en, te, message = THOUGHTS[date.today().toordinal() % len(THOUGHTS)]
    st.markdown(f'<div class="thought"><h2 style="color:#fff3a3!important">🌞 Today\'s Thought</h2><div class="quote">“{clean(en)}”</div><div class="telugu">{clean(te)}</div><div class="message">💡 {clean(message)}</div></div>', unsafe_allow_html=True)


def show_word(item):
    st.header(str(item['Word']).title())
    st.info('📖 English meaning: '+str(item['Meaning_English']))
    st.success('🌸 తెలుగు అర్థం: '+str(item['Meaning_Telugu']))
    a,b=st.columns(2)
    a.write('**Synonyms:** '+str(item['Synonyms']))
    b.write('**Antonyms:** '+str(item['Antonyms']))
    st.markdown(f'<div class="sentence en">🇬🇧 English example<br>{clean(item["Own_Sentence"])}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="sentence te">🌸 తెలుగు ఉదాహరణ<br>{clean(item["Own_Sentence_Telugu"])}</div>',unsafe_allow_html=True)
    forms={'Verb':'Verb_Base_Form','Noun':'Noun_Form','Adjective':'Adjective_Form','Adverb':'Adverb_Form'}
    cols=st.columns(4)
    for col,(label,key) in zip(cols,forms.items()):
        col.markdown('**'+label+'**')
        col.write(item[key] or 'Not provided')


PAGES=['🏠 Home','📖 Word Explorer','📊 Analytics','📈 Statistics','🧠 NLP Lab','🎯 Vocabulary Quiz','👩‍🏫 Study with Teacher','🛠 Data Quality']
if 'page' not in st.session_state:
    st.session_state.page=PAGES[0]

def go(page):
    st.session_state.page=page

csvs=sorted(BASE.glob('*.csv'))
with st.sidebar:
    st.title('📚 English for Begineers')
    st.caption('English–Telugu Vocabulary')
    for label in PAGES:
        st.button(label,key='side_'+label,on_click=go,args=(label,),use_container_width=True)
    st.divider()
    if csvs:
        preferred=[f for f in csvs if 'vocabulary' in f.name.lower()]
        chosen=st.selectbox('Vocabulary file',csvs,index=csvs.index(preferred[0]) if preferred else 0,format_func=lambda p:p.name)
    else:
        chosen=None
    st.caption(date.today().strftime('%d %B %Y'))

if chosen is None:
    st.error('Place your original 594-word CSV beside app.py, then refresh. No upload or CSV download is required.')
    st.code(str(BASE))
    st.stop()
try:
    df=read_csv(str(chosen))
except Exception as exc:
    st.error(f'Cannot read vocabulary: {exc}')
    st.stop()
if df.empty:
    st.warning('Your CSV has no vocabulary records.')
    st.stop()

page=st.session_state.page
if page==PAGES[0]:
    st.markdown('<div class="hero"><h1>📚 English for Beginners</h1><p>English–Telugu Vocabulary <br>Learn • Practice • Analyze • Improve</p></div>',unsafe_allow_html=True)
    st.write('')
    a,b,c=st.columns(3)
    a.metric('📚 Words',len(df))
    b.metric('🔤 Unique words',df.Word.nunique())
    c.metric('📖 Categories',df.Part_of_Speech.replace('',np.nan).nunique())
    st.subheader('👩‍🏫 Teacher Teaching • Student Learning')
    img=BASE/'teacher_student.png'
    if img.is_file():
        st.image(str(img),use_container_width=True)
    else:
        st.warning('Copy teacher_student.png beside app.py to show your selected teacher–student image.')
    st.subheader('🚀 Explore Features')
    for start in (1,4):
        cols=st.columns(3)
        for col,label in zip(cols,PAGES[start:start+3]):
            col.button(label,key='explore_'+label,on_click=go,args=(label,),use_container_width=True)
    thought_card()

elif page==PAGES[1]:
    st.title('📖 Word Explorer')
    q=st.text_input('Search word, English meaning, Telugu meaning or synonyms')
    pos=st.selectbox('Part of speech',['All']+sorted(x for x in df.Part_of_Speech.unique() if x))
    letter=st.selectbox('Starting letter',['All']+sorted(df.Initial.unique()))
    filtered=df.copy()
    if pos!='All': filtered=filtered[filtered.Part_of_Speech==pos]
    if letter!='All': filtered=filtered[filtered.Initial==letter]
    if q:
        fields=['Word','Meaning_English','Meaning_Telugu','Synonyms','Antonyms']
        filtered=filtered[pd.concat([filtered[k].str.contains(q,case=False,regex=False,na=False) for k in fields],axis=1).any(axis=1)]
    st.metric('Matching words',len(filtered))
    if filtered.empty: st.warning('No words found.')
    else:
        i=st.selectbox('Choose word',filtered.index.tolist(),format_func=lambda x:df.loc[x,'Word'])
        show_word(df.loc[i])

elif page==PAGES[2]:
    st.title('📊 Vocabulary Analytics')
    a,b,c=st.columns(3)
    a.metric('Total words',len(df));b.metric('Average word length',f'{df.Word_Length.mean():.2f}');c.metric('Average example length',f'{df.Sentence_Words.mean():.2f}')
    pos=df.Part_of_Speech.replace('','Unspecified').value_counts().reset_index()
    pos.columns=['Category','Count']
    letters=df.Initial.value_counts().sort_index().reset_index()
    letters.columns=['Letter','Count']
    a,b=st.columns(2)
    a.plotly_chart(px.bar(pos,x='Category',y='Count',color='Category',title='Parts of Speech',color_discrete_sequence=px.colors.qualitative.Bold),use_container_width=True)
    b.plotly_chart(px.bar(letters,x='Letter',y='Count',color='Count',color_continuous_scale='Plasma',title='A–Z Words'),use_container_width=True)
    a,b=st.columns(2)
    a.plotly_chart(px.histogram(df,x='Word_Length',nbins=20,title='Word Length',color_discrete_sequence=['#8b5cf6']),use_container_width=True)
    b.plotly_chart(px.histogram(df,x='Sentence_Words',nbins=20,title='Example Sentence Length',color_discrete_sequence=['#ec4899']),use_container_width=True)
    st.caption('These charts describe your curated CSV, not general English word frequency.')

elif page==PAGES[3]:
    st.title('📈 Statistical Analysis')
    variable=st.selectbox('Measure',['Word_Length','Sentence_Words'],format_func=lambda x:x.replace('_',' '))
    values=df[variable].dropna().astype(float)
    cols=st.columns(4)
    for col,name,value in zip(cols,['Mean','Median','Std deviation','Maximum'],[values.mean(),values.median(),values.std(),values.max()]):
        col.metric(name,f'{value:.2f}')
    st.dataframe(values.describe().to_frame('Value'),use_container_width=True)
    st.plotly_chart(px.histogram(df,x=variable,nbins=20,marginal='box',title='Distribution',color_discrete_sequence=['#9333ea']),use_container_width=True)
    if len(values)>1:
        sem=stats.sem(values)
        ci=stats.t.interval(.95,len(values)-1,loc=values.mean(),scale=sem) if sem>0 else (values.mean(),values.mean())
        st.info(f'Illustrative 95% t interval: {ci[0]:.2f} to {ci[1]:.2f}')
    groups=[g[variable].to_numpy(dtype=float) for _,g in df.groupby('Part_of_Speech') if len(g)>1]
    if len(groups)>1:
        try:
            result=stats.f_oneway(*groups)
            if np.isfinite(result.pvalue): st.write(f'Exploratory ANOVA: F={result.statistic:.3f}, p={result.pvalue:.4g}')
        except ValueError: st.warning('ANOVA not defined for these groups.')
    st.caption('Educational analysis of curated examples; statistical assumptions and sampling limitations apply.')

elif page==PAGES[4]:
    st.title('🧠 NLP Lab')
    st.write('NLP (Natural Language Processing) helps computers analyze human language. Try an English sentence below.')
    concepts={
        'Tokenization':'Splits text into tokens such as words and punctuation. Example: She is reading → She | is | reading.',
        'Lemmatization':'Finds a dictionary base form. Example: explained → explain.',
        'Part-of-Speech (POS) Tagging':'Predicts grammatical roles such as noun, verb, adjective and adverb.',
        'Stop Words':'Common words such as the, is, and that can be useful in grammar but sometimes excluded from search analysis.',
        'Named Entity Recognition':'Detects mentions of people, places, organizations, dates and similar entities.'}
    for concept,explanation in concepts.items():
        with st.expander('📘 '+concept): st.write(explanation)
    sentence=st.text_area('Enter your English sentence','The generous teacher patiently explained the difficult lesson.')
    if st.button('🔍 Analyze Sentence',type='primary'):
        if not sentence.strip(): st.warning('Please enter a sentence.')
        else:
            nlp,error=get_spacy()
            if nlp is None:
                st.warning('spaCy English model unavailable. Install it with: python -m spacy download en_core_web_sm')
                st.write('Basic tokens:',re.findall(r'\w+|[^\w\s]',sentence))
            else:
                doc=nlp(sentence)
                table=pd.DataFrame([{'Token':t.text,'Lemma':t.lemma_,'POS':t.pos_,'Explanation':pos_explanation(t.pos_),'Stop word':t.is_stop} for t in doc])
                st.dataframe(table,use_container_width=True,hide_index=True)
                counts=table.POS.value_counts().reset_index();counts.columns=['POS','Count']
                st.plotly_chart(px.bar(counts,x='POS',y='Count',color='POS',title='Grammatical Categories',color_discrete_sequence=px.colors.qualitative.Bold),use_container_width=True)
                if doc.ents:
                    st.write('**Named entities:**')
                    st.dataframe(pd.DataFrame([{'Entity':e.text,'Type':e.label_} for e in doc.ents]),hide_index=True)
                else: st.info('No named entities identified.')
                st.caption('POS tags and lemmas are model predictions, not a grammar checker or Telugu translator.')

elif page==PAGES[5]:
    st.title('🎯 Vocabulary Quiz')
    if 'quiz_index' not in st.session_state: st.session_state.quiz_index=random.randrange(len(df))
    if 'quiz_score' not in st.session_state: st.session_state.quiz_score=0
    if 'quiz_attempts' not in st.session_state: st.session_state.quiz_attempts=0
    if 'quiz_checked' not in st.session_state: st.session_state.quiz_checked=False
    i=st.session_state.quiz_index % len(df)
    item=df.iloc[i]
    st.metric('Score',f'{st.session_state.quiz_score} / {st.session_state.quiz_attempts}')
    st.subheader('What does this word mean?')
    st.header(item.Word)
    correct=item.Meaning_English
    pool=[v for v in df.Meaning_English.drop_duplicates().tolist() if v and v!=correct]
    choices=[correct]+random.Random(i).sample(pool,min(3,len(pool)))
    random.Random(i+42).shuffle(choices)
    answer=st.radio('Choose meaning',choices,key='answer_'+str(i))
    if st.button('Check Answer',disabled=st.session_state.quiz_checked):
        st.session_state.quiz_checked=True
        st.session_state.quiz_attempts+=1
        st.session_state.quiz_correct=(answer==correct)
        if st.session_state.quiz_correct: st.session_state.quiz_score+=1
    if st.session_state.quiz_checked:
        if st.session_state.quiz_correct: st.success('Correct! Well done.')
        else: st.error('Correct meaning: '+correct)
        st.write('తెలుగు:',item.Meaning_Telugu)
        st.write('Example:',item.Own_Sentence)
    if st.button('Next Word'):
        st.session_state.quiz_index=random.randrange(len(df))
        st.session_state.quiz_checked=False
        st.rerun()

elif page==PAGES[6]:
    st.title('👩‍🏫 Study with Teacher')
    img=BASE/'teacher_student.png'
    if img.is_file(): st.image(str(img),use_container_width=True)
    st.write('Select a word, study its meaning and examples, then write your own sentence.')
    i=st.selectbox('Choose lesson word',df.index.tolist(),format_func=lambda x:df.loc[x,'Word'])
    show_word(df.loc[i])
    practice=st.text_area('Write your own sentence using this word',key='practice_'+str(i))
    if st.button('Check Practice'):
        if not practice.strip(): st.warning('Write a sentence first.')
        elif re.search(r'\b'+re.escape(df.loc[i,'Word'])+r'\b',practice,re.I): st.success('You used the word! Compare your sentence with the example.')
        else: st.info('Try including the selected word.')
        st.caption('This checks word presence only, not grammatical correctness.')

elif page==PAGES[7]:
    st.title('🛠 Data Quality')
    required=['Word','Meaning_English','Meaning_Telugu','Own_Sentence','Own_Sentence_Telugu']
    st.dataframe(pd.DataFrame({'Column':required,'Empty values':[int(df[c].eq('').sum()) for c in required]}),hide_index=True,use_container_width=True)
    st.write('Duplicate words:',int(df.Word.str.lower().duplicated().sum()))
    st.dataframe(df[required],hide_index=True,use_container_width=True)
    st.caption('Data-quality checks do not verify translation accuracy.')

st.divider()
st.markdown('<p style="text-align:center;color:#6d28d9;font-weight:700">💜 LexiLens — Learn • Practice • Analyze • Improve</p>',unsafe_allow_html=True)
