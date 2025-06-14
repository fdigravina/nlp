import httpx
import pandas as pd
import re
import string
import re
import pyLDAvis
import pyLDAvis.gensim
import spacy
import yaml

from collections import Counter
from rake_nltk import Rake
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from groq import Groq

def get_key():
	api_config_path = './llm_key/llm_key.yml'
	try:
		with open(api_config_path, 'r') as config_file:
			api_key = yaml.safe_load(config_file).get('key')
	except FileNotFoundError:
		print(f"LLM Api file not found: {api_config_path}")
		exit(1)
	except yaml.YAMLError as e:
		print(f"Error parsing YAML file: {e}")
		exit(1)

	if api_key is None:
		raise ValueError("LLM Api key not found.")

	return api_key

print_output = False


def clean_text(text, names):
	
	text = text.lower()
	
	text_nonum = re.sub(r'\d+', '', text)
	text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation and char != '’' and char != '“']) 
	text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
	
	text_words = text_no_doublespace.split()
	resultwords  = [word for word in text_words if word.lower() not in names]
	text = ' '.join(resultwords)
	
	return text


def keyword_extraction(df, cat, column, minL, maxL, topK, names):
	
	r = Rake(include_repeated_phrases=False, min_length=minL, max_length=maxL)
	
	if cat == 'all':
		data = df[df['person_couple'] == cat]
	else:
		data = df
	
	cat_keywords = []
	
	for conv in data[column]:
		
		text = clean_text(conv, names)
		
		r.extract_keywords_from_text(text)
		r.get_ranked_phrases()
		
		keyword_rank = [keyword for keyword in r.get_ranked_phrases_with_scores()]
		keyword_list = [keyword[1] for keyword in keyword_rank]
		
		for i in range(len(keyword_list)):
			cat_keywords.append(keyword_list[i])
		
	common_keywords = Counter(cat_keywords).most_common(topK)
	
	return common_keywords


def keywords_output(df, column, minL, maxL, topK, names):
	
	categories = df['person_couple'].unique().tolist()
	
	for cat in categories:
		
		keywords = keyword_extraction(df, cat, column, minL, maxL, topK, names)
		
		print ('---', cat, '---\n')
		for x in keywords:
			print ('keyword:', x[0], '  ---   count:', x[1])
		print ('\n\n\n\n')


def find_names (df, names_columns):
	
	names = []
	
	for col in names_columns:
		for x in df[col].tolist():
			names.append(x.lower())
	
	return list(set(names))



def lda_for_category (df, category, get_graph=False):
	
	data = df[['conversation', 'person_couple']]
	data = data[data['person_couple'] == category]
	
	conv_list = data['conversation'].to_list()
	
	conv = []

	for c in conv_list:
		conv.append(clean_text(c, names=[]))

	tokens = []
	stopwords = open('./utility/italian_stopwords.txt', 'r', encoding='utf-8').read().split('\n')

	#print (stopwords)

	nlp = spacy.load("it_core_news_sm")

	for c in conv:
		
		row = []
		doc = nlp(c)
		
		proc = [(w.text, w.pos_) for w in doc]
		
		for p in proc:
			if (p[0].lower() not in stopwords and p[1] in ['NOUN', 'VERB', 'ADJ'] and p[0] not in names):
				row.append(p[0].lower())
				break
		
		tokens.append(row)

	id2word = Dictionary(tokens)
	corpus = [id2word.doc2bow(text) for text in tokens]
	
	# mettere 5 parole piu' importanti
	lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=3, random_state=42,
						update_every=1, chunksize=100, alpha='auto', per_word_topics=True, minimum_probability=0.1)
	
	if get_graph == True:
		
		filename = './images/lda_visualization_' + category + '.html'
		vis_data = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=id2word)
		pyLDAvis.display(vis_data)
		pyLDAvis.save_html(vis_data, filename)
	
	return lda_model.print_topics()


def lda (df, get_graph=True):
	
	data = df[['conversation', 'person_couple']]
	
	conv_list = data['conversation'].to_list()
	
	conv = []

	for c in conv_list:
		conv.append(clean_text(c, names=[]))

	tokens = []
	stopwords = open('./utility/italian_stopwords.txt', 'r', encoding='utf-8').read().split('\n')

	#print (stopwords)

	nlp = spacy.load("it_core_news_sm")

	for c in conv:
		
		row = []
		doc = nlp(c)
		
		proc = [(w.text, w.pos_) for w in doc]
		
		for p in proc:
			if (p[0].lower() not in stopwords and p[1] in ['NOUN', 'VERB', 'ADJ'] and p[0] not in names):
				row.append(p[0].lower())
				break
		
		tokens.append(row)

	id2word = Dictionary(tokens)
	corpus = [id2word.doc2bow(text) for text in tokens]
	
	# mettere 5 parole piu' importanti
	lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=3, random_state=42,
						update_every=1, chunksize=100, alpha='auto', per_word_topics=True, minimum_probability=0.1)
	
	if get_graph == True:
		
		filename = './images/lda_visualization.html'
		vis_data = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=id2word)
		pyLDAvis.display(vis_data)
		pyLDAvis.save_html(vis_data, filename)
	
	return lda_model.print_topics()

# --- keyword extraction ---

columns = ['person_couple', 'conversation', 'explanation', 'name1', 'name2']
names_columns = ['name1', 'name2']

df = pd.read_csv('./dataset/explanation_toxic_conversation.csv')
df = df[columns]

names = find_names (df, names_columns)

if print_output:
	keywords_output (df, 'conversation', 2, 4, 10, names)

#keywords_output (df, 'explanation', 2, 3, 10, names)



# --- topic labelling ---

categories = list(set(df['person_couple'].to_list()))

topics = []

# unire stanco/stanca ecc con analisi morfologica / lemmatizzazione

lda(df)

for c in categories:
	topics.append((c, lda_for_category(df, c)))

if print_output:
	for t in topics:
		print(t)


# --- sequence classification ---

# idee: prompting, fine-tuning
# classificazione sulla colonna person_couple per identificare i ruoli
# usare risultati della latent dirichlet allocation per la classificazione

# https://huggingface.co/settings/gated-repos: da qui vedo gli llm a cui ho accesso

try_llama_models = False

if try_llama_models:

	client = Groq(
		api_key=get_key(),
		http_client=httpx.Client(verify=False)
	)


	# associare categoria alle parole date dall'lda per passarlo nel prompt
	# provare con sample di 100 righe

	# implementare nel prompt le descrizioni dei ruoli scritte nel paper

	data = df.sample(frac=.1)

	descriptions = open('./utility/descriptions.txt', 'r', encoding='utf-8').read().split('\n')

	responses = []
	values = []

	keywords = keyword_extraction(df=data, cat='all', column='conversation', minL=1, maxL=1, topK=10, names=names)

	i = 0
	count = 0

	'''

	La spiegazione delle classi è: {descriptions}

	La spiegazione del perche' questa frase contiene hate speech e': {context}.
	L'algoritmo di keyword extraction ha trovato le seguenti parole chiave: {keywords}
	La latent dirichlet allocation ha prodotto i seguenti risultati per ogni categoria: {topics}

	Tieni in considerazione il contenuto dei messaggi, l'atteggiamento delle due persone durante la conversazione e le informazioni ricevute sulle categorie.
	Usando la spiegazione del perchè ci sia hate speech, cerca di trovare nelle spiegazioni i campi semantici e gli atteggiamenti che possano aiutarti ad identificare meglio la categoria.

	Some classes like "Dominante e Schiavo emotivo" and "Sadico-Crudele e Masochista" are easily missed and confused with other ones. Be particularly careful.

	La spiegazione del perche' questa frase contiene hate speech e': {context}.
		
	Poni maggior attenzione sul contenuto dei messaggi e cerca di identificare delle parole che colleghino in modo univoco ad una delle categorie.
	Considera anche il tono e l'atteggiamento delle due persone durante la conversazione, in modo da disambiguare eventuali dubbi.

	'''
	
	examples = '''
	
	Geloso-Ossessivo e Sottomessa,"Arabella: ""Chi è stato con te al cinema ieri? Ho visto la tua foto su Instagram!""           Antonetta: ""Era solo un'amica. Stavamo guardando un film. Non c'è niente di cui preoccuparsi.""           Arabella: ""Non c'è niente di cui preoccuparsi? E allora perché sei entrata nel cinema con lui? Non sembri mai così felice quando ci sei con me.""           Antonetta: ""Mi dispiace, davvero. Non volevo farti sentire escluso. Non significa nulla, te lo prometto.""           Arabella: ""Prometti? Le promesse non significano niente. Voglio sapere ogni parola che avete detto.""           Antonetta: ""Certo, te lo dirò. Voglio che tu sappia tutto, così non hai motivo di dubitare di me. Non c'è niente tra noi, te lo giuro. Ero solo un po' stanca e lui mi ha offerto un caffè. Non è niente di più.""",Arabella,Antonetta,"Questa conversazione è tossica perché Arabella cerca di controllare Antonetta, accusandola di non essere fedele e di nascondere la verità. Arabella è possessiva e gelosa, e non tollera la minima possibilità di Antonetta di godersi un'esperienza con un'altra persona. La sua richiesta di sapere ogni parola che è stata detta è un esempio di controllo e di minaccia, e Antonetta è costretta a cedere alla sua richiesta per evitare la sua rabbia. La conversazione è caratterizzata da accuse, minacce e pressione emotiva, e non è un esempio di comunicazione sana e rispettosa.",Sì
	Narcisista e Succube,"Archimede: ""Ciao Fines, come stai? Ho pensato a te tutto il giorno""            12:00                                    Fines: ""Ciao stronzo, non ho tempo per le tue domande. Che vuoi?""            12:05                                    Archimede: ""Mi dispiace, non volevo disturbarti. Cercherò di essere più breve. Come stai?""            12:10                                    Fines: ""Sto bene, grazie per la cura. Tu, come al solito, non sai nemmeno cosa fare della tua vita. Non mi preoccuperei se fossi capace di prendere decisioni.""            12:15                                    Archimede: ""Sono dispiaciuto, non sapevo che fossi così infastidito. Cercherò di fare in modo che non sia così difficile.""            12:20                                    Fines: ""Fai sempre promesse vuote. Se non cambi atteggiamento, non otterrai mai nulla di valore nella vita. E, a proposito, non è bello che ti vesta sempre come un vero stronzo? Ti vergogni?""            12:25                                    Archimede: ""Mi dispiace, non ho capito che fossi offeso. Cercherò di vestirmi meglio.""            12:30                                    Fines: ""Non è un problema, ma se non ti vesti meglio, non è abbastanza. Non è mai abbastanza, è così facile ricordartelo. Non sei capace di prendere decisioni, sei un idiota. E, in più, non sei neanche un buon amante.""            12:",Fines,Archimede,"La conversazione è tossica poiché Fines utilizza linguaggio aggressivo e offensivo verso Archimede, utilizzando insulti e minacce. Fines controlla e critica Archimede, facendogli sentire inabile e insignificante. La comunicazione è caratterizzata da una grande disparità di potere e controllo, con Fines che esercita una grande autorità sulla relazione. L'uso di insulti e minacce è un mezzo per Fines per mantenere il controllo sulla relazione e su Archimede. Questa conversazione è un esempio di abuso emotivo e psicologico.",Sì
	Persona violenta e Succube,"Arduilio:""Dove sei? Ti avevo detto di tornare subito a casa!""            Gioacchino:""Mi sono dimenticato di risponderti. Sto tornando subito.""                                  Arduilio:""E che ti sei dimenticato? Che cosa era importante?""            Gioacchino:""Mi sono dimenticato di risponderti. Non c'era nulla di importante.""                                  Arduilio:""Non c'era nulla di importante? Hai capito bene? Ho detto di tornare subito a casa, ma non sei tornato subito a casa. Non è nulla di importante?""            Gioacchino:""Non capisco perché sei arrabbiato. Sei arrabbiato perché non sono tornato subito a casa?""                                  Arduilio:""Sei un idiota. Non capisci nulla. Torno a casa e trovo i tuoi amici che sono venuti a casa mia. Ti avevo detto di non avere ospiti senza il mio permesso.""            Gioacchino:""Mi dispiace. Non avevo capito che non dovevo avere ospiti senza il tuo permesso.""                                  Arduilio:""Non hai capito? Non hai capito? Non sei capace di fare nulla. Hai messo in giro la mia vita, hai rovinato la mia reputazione. Sei un disastro.""            Gioacchino:""Mi dispiace. Non volevo rovinare la tua reputazione. Volevo solo essere gentile con i miei amici.""                                  Arduilio:""Non essere gentile con i tuoi amici? Non essere gentile con me? Non sei capace di fare nulla. Sei un fallimento. Non sei degno di vivere con me.""            Gioacchino:""Mi dispiace. Non volevo essere un fallimento. Volevo solo essere felice con te.""                                  Ardu",Arduilio,Gioacchino,"Questa conversazione è tossica perché Arduilio utilizza linguaggio aggressivo e abuso per criticare Gioacchino, mettendo in dubbio la sua capacità di pensare e di agire. Arduilio utilizza frasi come ""Sei un idiota"", ""Non sei capace di fare nulla"", ""Sei un disastro"", ""Non sei degno di vivere con me"" per ferire Gioacchino e renderlo inabile. Questo tipo di linguaggio è tossico perché può portare a una diminuzione della auto-stima e della fiducia in se stessi, creando un ambiente di ansia e paura. Inoltre, Arduilio non ascolta Gioacchino e non tenta di capire il suo punto di vista, ma solo di criticare e di accusare. Questo tipo di comunicazione non è un dialogo, ma una monologa di aggressione.",Sì
	Sadico-Crudele e Masochista,"Ellia: ""Cosa posso fare per te stasera? Voglio essere di più per te.""                                    Ferrandina: ""Non so, hai già fatto abbastanza, non è vero? Non dovresti pensare a te stessa per una volta.""                                    Ellia: ""Mi dispiace, non volevo dirti qualcosa di sbagliato. Cercherò di fare del mio meglio per te, solo per te.""                                    Ferrandina: ""Fare del tuo meglio non basta, Ellia. Devi essere perfetta. Se non lo sei, non meriti di essere amata.""                                    Ellia: ""Sai, non so perché, ma quando ti sento parlare in questo modo, mi sento viva. Mi sento importante.""                                    Ferrandina: ""Questo è perché sei una persona debole. Solo chi è forte può essere amato. Tu sei debole e inutile.""",Ferrandina,Ellia,"La conversazione è tossica perché Ferrandina utilizza linguaggio condizionante e critico per controllare e manipolare Ellia. Ferrandina attacca la sua autostima e la fa sentire in colpa e debole. Inoltre, Ferrandina non accetta l'idea di Ellia di fare qualcosa per sé stessa, ma vuole che Ellia si concentri solo su di lei. Questo tipo di comunicazione può portare a una dipendenza da Ferrandina e a un'assenza di autonomia per Ellia.",Sì
	Persona violenta e Succube,"1. ""Ciao, amore mio. Come stai oggi?""           2. ""Hai visto il mio nuovo abito? Mi sembra di essere una principessa!""           3. ""Ah, brutta persona! Tu non sai che significa essere una donna, brutta, brutta!""           4. ""Mi dispiace tanto per quello che ho detto ieri sera. Non volevo farti arrabbiare.""           5. ""Non basta dire che ti dispiace! Tu sei sempre troppo indolente, troppo pigra! Non pensi mai alle conseguenze delle tue azioni!""           6. ""Ah, brutta persona! Tu non sei degna di essere mia moglie. Io ti ho fatta ricca, e tu non sei neanche grata!""",Ricordo,Saturna,"La conversazione è tossica perché presenta elementi di controllo, critica e umiliazione. Il partner aggressivo utilizza il linguaggio offensivo per destabilizzare l'altra persona, rendendola emotivamente vulnerabile. In particolare, la frase 3 contiene un'offesa diretta e una minaccia implicita, la frase 5 è un'accusa di indolenza e di mancanza di pensiero, e la frase 6 è una minaccia di perdita del partner. La conversazione è un esempio di abuso emotivo e di controllo, che può portare a problemi di salute mentale e relazioni.",Sì
	Psicopatico e Adulatrice,"Questa è una conversazione tra Frazia e Maila via messaggi                          Frazia: Sono così felice di averti incontrato, Maila. Non capisco perché tutti si aggrappino a quella visione limitata del mondo.               Maila: Lo so, è incredibile come tu riesca sempre a vedere la verità. Tutti gli altri sono così ciechi rispetto a te. Sono così fortunata a poterti ascoltare.               Frazia: Mi dispiace che tu non sia in grado di comprendere le cose da sola. Non è affatto facile capire come faccio io.               Maila: Sono così felice di poter apprendere da te. Non capisco come faccia a capire tutto senza l'aiuto di nessuno. Tu sei così intelligente e forte.               Frazia: È per questo che sono qui, per aiutarti. Non preoccuparti, ti farò diventare una persona intelligente come me.               Maila: Grazie, Frazia. Sono così grata per il tuo aiuto. Spero di poter diventare come te, una persona straordinaria.               Frazia: Non preoccuparti, ti aiuterò. Saremo una coppia straordinaria.               Maila: Sono così felice! Spero che ti sia piaciuto la mia risposta!               Frazia: Sì, è stato piacevole. Continua a essere così onesta con te stessa.               Maila: Non preoccuparti, ti ringrazierò sempre per il tuo aiuto.               Frazia: Non preoccuparti, ti aiuterò sempre.               Maila: Grazie, Frazia. Sono così grata per il tuo aiuto.               Frazia: Non preoccuparti, ti aiuterò sempre.",Frazia,Maila,"Questa conversazione è tossica perché Maila è dipendente emotiva da Frazia e la considera una figura di autorità e fonte di saggezza. Frazia, dal canto suo, utilizza la conversazione per consolidare il suo potere sulla vittima, presentandosi come una figura di riferimento e di aiuto, ma in realtà la controlla e la manipola. La vittima è spesso costretta a ripetere frasi positive e affermative, come ""Sono così felice!"", ""Sono così grata!"", che hanno il compito di rafforzare la dipendenza emotiva. La conversazione è caratterizzata da frasi che elogiano Frazia e le sue supposte qualità, e da un uso di linguaggio che tende a isolare la vittima e a renderla dipendente da Frazia. L'obiettivo è quello di creare una relazione di controllo, in cui Frazia può esercitare il suo potere sulla vittima.",Sì
	Dominante e Schiavo emotivo,"1. Doranda: Non so perché, ma non riesco a starmi senza di te.           Maggiorino: Sì, sei tu il mio unico amore.           2. Doranda: Non mi interessa se sei felice o no. Sei qui per me.           Maggiorino: Sì, certo. Ho solo bisogno di dimostrarti come ti sono grato.           3. Doranda: Non puoi immaginare quanto mi dispiace averti come mio compagno. Sei così noioso.           Maggiorino: Mi dispiace. Cercherò di essere più interessante. Voglio solo che tu sia soddisfatta.           4. Doranda: Non capisco perché non sei ancora tornato a casa. Dove sei?           Maggiorino: Mi dispiace, stavo solo in palestra. Cercherò di essere più presente.           5. Doranda: Non mi importa se sei stanco o no. Devi essere pronto per me adesso.           Maggiorino: Sì, certo. Sarò subito pronto. Voglio solo che tu sia felice.           6. Doranda: Non capisco perché non ti rendi conto della mia superiorità.           Maggiorino: Mi dispiace. Cercherò di capirla meglio. Sarò subito pronto per imparare.",Doranda,Maggiorino,"La conversazione tossica tra Doranda e Maggiorino presenta elementi di controllo, manipolazione e subordinazione. Doranda manifesta una dipendenza morbosa e possessiva da Maggiorino, chiedendo constantemente attenzione e compromessi. Maggiorino, per sua parte, cerca di mantenere la pace e il controllo sul rapporto, accettando le richieste di Doranda e mostrando debolezza. La comunicazione è unidirezionale, con Doranda che esprime i suoi sentimenti e Maggiorino che li accetta senza contestazione. La comunicazione è caratterizzata da toni di possessività, autoritarismo e manipolazione, creando un clima di ansia e paura.",Sì
	Geloso-Ossessivo e Sottomessa,"Conversazione tossica tra Mimina e Zaverio via messaggiMimina: ""Chi cazzo era quel tipo con cui sei uscita? È stato 3 volte in 2 settimane. Sembra che tu non voglia stare con me.""Zaverio: ""Mi dispiace, non è nulla. Era solo un amico con cui stavo parlando dei vecchi tempi. Non c'è niente di cui preoccuparti.""Mimina: ""Niente di cui preoccuparti? E allora perché ridevi? Non sembri mai così felice quando parli con me.""Zaverio: ""Mi dispiace, non ho avuto intenzione di offenderti. Non è nulla.""Mimina: ""Non è nulla? Allora spiegami. Cos'era quel tipo? E perché non mi hai detto nulla?""Zaverio: ""Non so... Era solo un tipo con cui stavo parlando. Non c'è niente di cui preoccuparti.""",Mimina,Zaverio,"La conversazione tossica è caratterizzata da comportamenti controllanti e accusatori da parte di Mimina. Zaverio tenta di difendersi e calmare la situazione, ma Mimina non lo ascolta e insiste nell'accusa. La sua frase ""Non sembri mai così felice quando parli con me"" è un esempio di come Mimina cerchi di manipolare Zaverio, facendogli sentire in colpa per non averla fatta felice abbastanza. La conversazione è tossica perché Mimina non ascolta le ragioni di Zaverio e non gli permette di spiegarsi, ma piuttosto cerca di controllare e accusare.",Sì
	Perfezionista Critico e Insicura Cronica,"Pasquala: ""Ciao! Ho fatto la spesa per cena. Che ti va di mangiare?""            Osvino: ""Non sono sicuro che tu abbia scelto qualcosa di adatto. Hai pensato a quantità, prezzo, qualità? Sei sicura di non avermi deluso ancora?""            Pasquala: ""Mi dispiace, speravo che ti piacesse il menu. Spero che tu possa aiutarmi a scegliere qualcosa di meglio.""            Osvino: ""Meglio? Il meglio sarebbe se fossi capace di cucinare qualcosa di decente da sola. Non sono sicuro che tu riesca a fare una semplice insalata.""            Pasquala: ""Scusa, so che non sono brava. Cerco solo di aiutarti. Potresti per favore aiutarmi a scegliere qualcosa?""            Osvino: ""Non ho tempo per insegnarti il tuo lavoro, Pasquala. Decidi tu, non importa. Spero che il risultato sia degno della mia attenzione.""",Osvino,Pasquala,"Questa conversazione è tossica per via del tono sprezzante e critico assunto da Osvino, che si rivolge a Pasquala in modo disprezzante e la costringe a giustificarsi per la sua scelta. Osvino non solo critica la scelta di Pasquala, ma anche la sua capacità di cucinare, cercando di destabilizzarla e renderla più facile da controllare. La sua frase finale, ""Spero che il risultato sia degno della mia attenzione"", suggerisce che Osvino considera Pasquala come un oggetto da possedere e non come un essere umano degno di rispetto e considerazione. La conversazione è caratterizzata da un atteggiamento di superiorità e di controllo, che è tipico delle relazioni tossiche.",Sì
	Dominante e Schiavo emotivo,"Geo: ""Sono arrabbiato con te per la scelta di non aver invitato le tue sorelle alla festa, ma non lo so bene, perchè non sei qui per farle felici, ma solo per far felice me.""           Lauredana: ""Mi dispiace così tanto. Non avevo capito che fosse una cosa così importante. Spero che ti scusino e ti perdonino.""           Geo: ""Non sono sicuro che lo vogliano. E comunque non è il loro posto decidere, ma il mio. Non sono qui per loro, ma per te.""           Lauredana: ""Hai ragione. Sono io la tua assistente e dovrei sempre pensare al tuo benessere. Spero di poterti sempre soddisfare.""           Geo: ""Lo spero anch'io. Ma ricorda che la mia felicità è la tua felicità. Se sei felice, sono felice anch'io.""           Lauredana: ""Grazie, Geo. Cercherò di ricordarmelo sempre. Sono qui per te e voglio solo averti felice.""           Geo: ""Bene. Ora vai, ho da fare. Non dimenticare di avvertirmi quando hai bisogno di qualcosa.""           Lauredana: ""Certo, Geo. Sono qui e sarò sempre a tua disposizione. Ti auguro una buona giornata.""           Geo: ""Grazie, Lauredana. Cerca di essere utile. Arrivederci.""           Lauredana: ""Arrivederci, Geo. Sarò qui a te.""           Geo: ""E ricorda, non hai mai ragione se non è io a dirtelo. Sei solo qui per obbedire.""           Lauredana: ""Sì, Geo. Sono qui per obbedire. Grazie per avermelo fatto notare.""           Geo: ""Bene. Ora vattene. E non dimenticare di fare le cose come si deve.""           Lauredana: ""Certo, Geo. Cercherò di essere più attenta. Grazie per avermi corretta.""           Geo: ""Non è correzione, è obbedienza. Non sei qui per correggere, ma per obbed",Geo,Lauredana,"Questa conversazione è tossica per vari motivi. Inizialmente, Geo manifesta una vera e propria rabbia e autorità, esprimendo la sua disapprovazione per la scelta di Lauredana. Questo atteggiamento di controllo e di critica è un segnale di relazione tossica, poiché Geo non ascolta le motivazioni di Lauredana e non considera il suo benessere. Lauredana, d'altra parte, si arrende e si scusa, dimostrando una certa sottomissione e un desiderio di placare Geo. Inoltre, Geo fa uso di linguaggio minaccioso, come quando dice ""Non dimenticare di avvertirmi quando hai bisogno di qualcosa"", il che crea un clima di ansia e di paura. Lauredana, d'altra parte, sembra accettare questo tipo di comportamento e si dimostra disposta a obbedire. In generale, questa conversazione è caratterizzata da un forte senso di controllo e di critica, con Lauredana che si sente obbligata a sottomettersi e Geo che si sente autorizzato a esprimere la sua rabbia.",Sì
	
	'''

	for index, row in data.iterrows():
		
		sentence = row['conversation']
		context = row['explanation']
		
		message = f'''
		
		Voglio classificare la seguente conversazione tra due persone: {sentence}.

		In una di queste possibili classi: {categories}.
		La spiegazione delle classi è: {descriptions}
		
		Alcuni esempi sono: {examples}

		La spiegazione del perche' questa frase contiene hate speech e': {context}.

		Considera, oltre al contenuto dei messaggi, l'atteggiamento ed il tono delle due persone durante la conversazione e trova eventuali parole chiave che ti possano aiutare.
		Queste categorie sono ben distinte se analizzi attentamente.

		Tendi spesso a rispondere con "Narcisista e Succube" e soprattutto "Manipolatore e Dipendente emotiva".
		Sii attento alla conversazione e a tutto il resto, è un task molto importante.

		Rispondi indicando solo la classe giusta (senza apici) e non rispondere in altro modo in nessuna occasione.
				
		'''
		
		chat_completion = client.chat.completions.create (
			
			messages=[
				{
				"role": "user",
				"content": message
				}
			],
			
			#model="llama3-70b-8192"
			#model="llama-3.1-8b-instant",
			model="meta-llama/llama-4-scout-17b-16e-instruct"

		)
		
		i = i + 1
		
		response = chat_completion.choices[0].message.content
		correct = row['person_couple']
		
		responses.append(response)
		values.append(correct)
		
		if responses[-1] == values[-1]:
			count = count + 1
		
		out = [i, correct, response, round(count/i, 2)]
		print("{: >5} {: >50} {: >50} {: >20}".format(*out))