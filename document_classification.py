import xml.etree.ElementTree as ET
import os,math
from collections import Counter
import pickle
from liblinearutil import *

def preprocess(raw_text_file, corenlp_output):
        command = "java -cp /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -file " + str(raw_text_file) + " -outputDirectory " + str(corenlp_output)
        os.system(command)

def get_all_files_absolute(directory):
        abs_filepath = []
        for dirpath,dirnames,files in os.walk(directory):
                for filename in files:
                        filec = os.path.join(dirpath,filename)
                        abs_filepath.append(filec)
        return abs_filepath

def extract_top_words(xml_directory):
    	top_words = [];
	#Get all xml files
	all_words = []
	xml_files = get_all_files_absolute(xml_directory)
	#Process each xml file
	for each in xml_files:
		tree = ET.parse(each)
		root = tree.getroot()
		for token in root.iter('token'):
			word = token.find('word').text.lower()
			all_words.append(word)
	frequency = Counter(all_words)
	#print frequency
	top_words = sorted(frequency, key=frequency.get, reverse=True)[0:2000]
	#print frequency
	#print top_words


    	return top_words;

def map_unigrams(xml_filename, top_words):
	vector =  [];
	xml_words = []
	tree = ET.parse(xml_filename)
        root = tree.getroot()
       	for token in root.iter('token'):
        	word = token.find('word').text.lower()
       	        xml_words.append(word)

	for each in top_words:
                if each in xml_words:
                        vector.append(1)
                else:
                        vector.append(0)
	return vector 

#Cosine similarity function

def cosine_similarity(X,Y):

	if(all(x == 0 for x in X) or all(y == 0 for y in Y)):
		return 0
        dot_product = 0
        deno1 = 0
        deno2 = 0
        for i in range(len(X)):
                dot_product = dot_product + float(X[i])*float(Y[i])
                deno1 = deno1 + float(X[i])*float(X[i])
                deno2 = deno2 + float(Y[i])*float(Y[i])
        sq1 = math.sqrt(deno1)
        sq2 = math.sqrt(deno2)
        dot_product = dot_product + 0.0
        deno = sq1*sq2
        cosine = float(dot_product/deno)
        return cosine

def extract_similarity(top_words):
	f = open('/project/cis/nlp/tools/word2vec/vectors.txt')
        #f = open('temp.txt')
        wordvec = {}
        for line in f:
                w = line.split()
                if(w[0] in top_words):
                        wordvec[w[0]] = w[1:]
                        #print wordvec
        f.close()
        sim_mat = {}
        for word1 in top_words:
                sim_mat[word1] = {}
                for word2 in top_words:
                        if(word1 == word2):
                                sim_mat[word1][word2] = 1.0
                                continue
                        if((word1 not in wordvec) or (word2 not in wordvec)):
                                continue
                        if(word2 in sim_mat):
                                if(word1 in sim_mat[word2]):
                                        similarity = sim_mat[word2][word1]
                                else:
                                        similarity = 0
                        else:
                                similarity = cosine_similarity(wordvec[word1], wordvec[word2])
                        if(similarity != 0):
                                sim_mat[word1][word2] = similarity

        return sim_mat

def map_expanded_unigrams(xml_filename, top_words, similarity_matrix) :
	vec = [] 
	feature_vec = map_unigrams(xml_filename, top_words)
	vec = []
	i = 0
	for each in feature_vec:
		if(each == 1):
			vec.append(1.0)
		else:
			z_word = top_words[i]
			index = 0
			maxi = -1000
			for each1 in feature_vec:
				if(each1 == 1):
					nz_word = top_words[index]
					temp = similarity_matrix[z_word]
					if nz_word in temp:
						answer = similarity_matrix[z_word][nz_word]
					else:
						answer = 0
					if(answer > maxi):
						maxi = answer
				index = index + 1
			vec.append(maxi)
		i = i + 1
	
	return vec;


def extract_top_dependencies(xml_directory):
	top_dep =  [];
	all_dep = {}
	
	xml_files = get_all_files_absolute(xml_directory)
        #Process each xml file
        for each in xml_files:
                tree = ET.parse(each)
                root = tree.getroot()
                for token in root.iter('dep'):
			relation = token.attrib['type']
			governor  = token.find('governor').text.lower()
			dependent = token.find('dependent').text.lower()
			if((relation,governor,dependent) in all_dep):
				all_dep[(relation,governor,dependent)] =all_dep[(relation,governor,dependent)] + 1
			else:
				all_dep[(relation,governor,dependent)] = 1

        top_dep = sorted(all_dep, key=all_dep.get, reverse=True)[0:2000]
	return top_dep;

def map_dependencies(xml_file, dep_list):
	vec = []
	all_dep = []
	tree = ET.parse(xml_file)
        root = tree.getroot()
        for token in root.iter('dep'):
        	relation = token.attrib['type']
                governor  = token.find('governor').text.lower()
                dependent = token.find('dependent').text.lower()
		all_dep.append((relation,governor,dependent))
	for each in dep_list:
		if(each in all_dep):
			vec.append(1)
		else:
			vec.append(0)
	return vec;

def extra_extract_prod_rs(xml_file):
	tree = ET.parse(xml_file)
        root = tree.getroot()
	stack = []
	dic = {}
	prod_rules = []
	for p_tree in root.iter('parse'):
		p_tree = p_tree.text
		p_tree = p_tree.split()
		for item in p_tree:
			if item[0] == '(':
				if item[1:] not in stack:
					stack.append(item[1:])
					dic[item[1:]] = []
				else:
					i = 1
					while(True):
						temp_item = item[1:] + '_' + str(i) + '_FLAGGED'
						if temp_item not in stack:
							break
						i = i + 1
					stack.append(temp_item)
					dic[temp_item] = []
			elif item[len(item) - 1] == ')':
				for i in range(0, len(item)):
					if(item[i] == ')'):
						break
				length = len(item)
				for j in range(i, length):
					top_item = stack.pop()
					if(len(stack) != 0 ):
						prev = stack[len(stack) - 1]
						dic[prev].append(top_item)
					rules = dic[top_item]
					dic.pop(top_item, None)
					if (len(rules) != 0):
						temp = top_item.split('_')
						top_item = temp[0]
						rule = top_item
						
						for right in rules:
							temp = right.split('_')
							right = temp[0]
							rule = rule + '_' + right
						prod_rules.append(rule)

	return prod_rules

def extract_prod_rules(xml_directory):
	top_prod = [];
	prod = []
	xml_files = get_all_files_absolute(xml_directory)
	for each in xml_files:
		prod = prod + extra_extract_prod_rs(each)
	frequency = Counter(prod)	
	
	
	top_prod = sorted(frequency, key=frequency.get, reverse=True)[0:2000]
	return top_prod;

def map_prod_rules(xml_file, rule_list):
	vec = [];
	prod_rules = extra_extract_prod_rs(xml_file)
	for rule in rule_list:
		if(rule in prod_rules):
			vec.append(1)
		else:
			vec.append(0)
			
	return vec;

def process_corpus( xml_root, top_words, similarity_matrix, top_dependencies, syntactic_prod_rules ) :
	if(xml_root.find('test') != -1):
		bin_lexical = open('test_1.txt', 'w')
		lexical_expansion = open('test_2.txt', 'w')
		bin_dependency = open('test_3.txt', 'w')
		bin_rules = open('test_4.txt', 'w')
		all_except_expanded = open('test_5.txt', 'w')
	else:
		bin_lexical = open('train_1.txt', 'w')
		lexical_expansion = open('train_2.txt', 'w')
		bin_dependency = open('train_3.txt', 'w')
		bin_rules = open('train_4.txt', 'w')
		all_except_expanded = open('train_5.txt', 'w')

	files = get_all_files_absolute(xml_root)
	for each in files:
		unigram_vector = map_unigrams(each, top_words)
		expanded_unigram_vector = map_expanded_unigrams(each, top_words, similarity_matrix)
		dependency_vector = map_dependencies(each, top_dependencies)
		prod_rules_vector = map_prod_rules(each, syntactic_prod_rules)
		all_except_expanded_i = 1

		all_except_expanded.write(os.path.basename(each))
		bin_lexical.write(os.path.basename(each))
		cnt = 1
		for element in unigram_vector:
			if(element > 0):
				bin_lexical.write(' ' + str(cnt) + ':' + str(element))
				all_except_expanded.write(' ' + str(all_except_expanded_i) + ':' + str(element))
			all_except_expanded_i = all_except_expanded_i + 1
			cnt = cnt + 1
		bin_lexical.write('\n')

		lexical_expansion.write(os.path.basename(each))
		cnt = 1
		for element in expanded_unigram_vector:
			if(element > 0):
				lexical_expansion.write(' ' + str(cnt) + ':' + str(element))
			cnt = cnt + 1
		lexical_expansion.write('\n')

		bin_dependency.write(os.path.basename(each))
		cnt = 1
		for element in dependency_vector:
			if(element > 0):
				bin_dependency.write(' ' + str(cnt) + ':' + str(element))
				all_except_expanded.write(' ' + str(all_except_expanded_i) + ':' + str(element))
			all_except_expanded_i = all_except_expanded_i + 1
			cnt = cnt + 1
		bin_dependency.write('\n')

		bin_rules.write(os.path.basename(each))
		cnt = 1
		for element in prod_rules_vector:
			if(element > 0):
				bin_rules.write(' ' + str(cnt) + ':' + str(element))
				all_except_expanded.write(' ' + str(all_except_expanded_i) + ':' + str(element))
			all_except_expanded_i = all_except_expanded_i + 1
			cnt = cnt + 1
		bin_rules.write('\n')
		all_except_expanded.write('\n')


	lexical_expansion.close()
	bin_dependency.close()
	bin_rules.close()
	all_except_expanded.close()

def create_feature_domain_file(feature_set, domain, train_test):
	filename = open(domain + '_' + train_test + '_' + feature_set + '.txt', 'w')
	input_file = open(train_test + '_' + feature_set + '.txt', 'r')

	input_file = input_file.readlines()

	for each in input_file:
		vector = each.split()
		initial = vector[0]
		if(initial.startswith(domain)):
			filename.write('1')
			for i in range(1, len(vector)):
				filename.write( ' ' + vector[i])
			filename.write('\n')
		else:
			filename.write('-1')
			for i in range(1, len(vector)):
				filename.write( ' ' + vector[i])
			filename.write('\n')
	filename.close()

def generate_liblinear_files():
	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Computers', 'train')
	
	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Finance', 'train')

	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Health', 'train')

	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Research', 'train')

	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Computers', 'test')
	
	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Finance', 'test')

	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Health', 'test')

	for each in range(1, 6):
		create_feature_domain_file(str(each), 'Research', 'test')

def get_precision(pre_labels, act_labels, label):
	num = 0
	deno = 0
	index = 0
	for each in pre_labels:
		if(each == label):
			deno = deno + 1
			if(each == act_labels[index]):
				num = num + 1
		index = index + 1
	if(deno == 0):
		return 0
	precision = float(num)/float(deno)
	return precision

def get_recall(pre_labels, act_labels, label):
	num = 0
	deno = 0
	index = 0
	for each in act_labels:
		if(each == label):
			deno = deno + 1
			if(pre_labels[index] == each):
				num = num + 1
		index = index + 1
	if(deno == 0):
		return 0
	recall = float(num)/float(deno)
	return recall

def get_fmeasure(precision, recall):
	if((precision + recall) == 0):
		return 0
	fmeasure = 2*precision*recall/(precision+recall)
	return fmeasure

def get_weight(f,key):
	num = 0
	deno = 0
	lines = open(f).readlines()
	for line in lines:
		if line.split()[0]==key:
			num+=1
		deno+=1
	if(deno == 0):
		return 0
	return float(num)/float(deno)

def run_classifier(train_file, test_file):
	output_tuple = ();
	y,x = svm_read_problem(train_file) 
	weight2 = get_weight(train_file,"1")
	weight1 = 1.0 - weight2
	model = train(y,x,'-s 0 -w1 '+ str(weight1) +' -w-1 '+ str(weight2))
    	y, x = svm_read_problem(test_file)
    	p_labs, p_acc, p_vals = predict(y,x, model,'-b 1')
    	p_precision = get_precision(p_labs,y,1)
    	p_recall = get_recall(p_labs,y,1)
    	p_fmeasure = get_fmeasure(p_precision,p_recall)
    	n_precision = get_precision(p_labs,y,-1)
	n_recall = get_recall(p_labs,y,-1)
    	n_fmeasure = get_fmeasure(n_precision,n_recall)
    	measures = (p_precision, p_recall, p_fmeasure, n_precision, n_fmeasure, n_recall, p_acc[0])
    	prob_list = []	
    	for sublist in p_vals:
		prob_list.append(sublist[0])
    	output_tuple = (p_labs, measures,prob_list)
	return output_tuple;

def generate_results():
	f = open('results.txt', 'w')
	domains = ['Computers', 'Finance', 'Health', 'Research']
	for each in domains:
		for i in range(1,6):
			output_tuple = run_classifier(each + '_train_' + str(i) + '.txt', each + '_test_' + str(i) + '.txt')
			measures = output_tuple[1]
			for measure in measures:
				f.write(str(measure) + ' ')
			f.write(each + ':' + str(i) + '\n')
	f.close()


def classify_documents(health_prob, computers_prob, research_prob, finance_prob):
	p_labels = [];
	for i in range(0,len(health_prob)):
		if ((health_prob[i] >= computers_prob[i]) and (health_prob[i] >= research_prob[i]) and (health_prob[i] >= finance_prob[i])):
			p_labels.append("Health")
		elif ((computers_prob[i] >= health_prob[i]) and (computers_prob[i] >= research_prob[i]) and (computers_prob[i] >= finance_prob[i])):
			p_labels.append("Computers")
		elif ((research_prob[i] >= health_prob[i]) and (research_prob[i] >= computers_prob[i]) and (research_prob[i] >= finance_prob[i])):
			p_labels.append("Research")
		else:
			p_labels.append("Finance")
	return p_labels

def generate_classify_results():
	f = open("temp_results.txt","w")

	domains = ["Health","Computers","Research","Finance"]
    	d_prob = []
    	for each in domains:
        	tup = run_classifier(each +"_train_"+ "5" +".txt",each +"_test_"+ "5" +".txt")
        	d_prob.append(tup[2])
	p_labels = classify_documents(d_prob[0], d_prob[1], d_prob[2], d_prob[3])
	#print len(p_labels)
	cnt = 0
    	actual_labels = get_all_files_absolute("test_data")
    	total = len(actual_labels)
	#print total
    	for i in range(0,total):
        	if p_labels[i] in actual_labels[i]:
            		cnt+=1
	#print cnt
    	accuracy = cnt/(total*1.0)
	accuracy = accuracy*100
    	f.write(str(accuracy))
	f.close()

def save_object(obj, filename):
     filehandler = open(filename, 'wb')
     pickle.dump(obj, filehandler, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
	file_handler = open(filename,'rb')
	object_file = pickle.load(file_handler)
	file_handler.close()
	return object_file


if __name__ == "__main__":
   	"""
	#input_directory = "/home1/c/cis530/hw3/xml_data"
	#files = get_all_files_absolute("/home1/c/cis530/hw3/test_data")
	#for f in files:
	#	print f
	#input_directory = "test"
	
	#xml_directory = "/home1/c/cis530/hw3/xml_data"
	
	#top_words = extract_top_words(xml_directory)
	#print top_words
	#similarity_matrix = extract_similarity(top_words)
	#print similarity_matrix
	# map_unigrams("Research_2005_01_02_1638819.txt.xml", top_words)
	#vec = map_expanded_unigrams("Research_2005_01_02_1638819.txt.xml", top_words, similarity_matrix)
	#print vec
	#top_dep = extract_top_dependencies(xml_directory)
	#print top_dep
	#dep_list = [('det', 'calendar', 'a'),('hello','shivam','dalal')]
	#print map_dependencies("Computers_and_the_Internet_2005_01_06_1639801.txt.xml", dep_list)
	#rule_list = extract_prod_rules(xml_directory)
	#vec = map_prod_rules("Computers_and_the_Internet_2005_01_06_1639801.txt.xml", rule_list)
	#print vec
	#preprocess("input.txt","/home1/s/shivamda/CL/hw3/test_data")
	#process_corpus( xml_directory, top_words, similarity_matrix, top_dep, rule_list ) 
	
	cwd = os.getcwd()
	
	if os.path.isfile(cwd + '/rules.txt'):
		print 'Loading from rules.txt'
		prod_rules = load_object(cwd + '/rules.txt')
	else:
		print 'Saving to rules.txt'
		#'/home1/c/cis530/hw3/xml_data'
		prod_rules = extract_prod_rules('/home1/c/cis530/hw3/xml_data')
		save_object(prod_rules, cwd + '/rules.txt')
	#print '##### PROD RULES #####'
	#print prod_rules
	
	if os.path.isfile(cwd + '/topwords.txt'):
                print 'Loading from topwords.txt'
                top_words = load_object(cwd + '/topwords.txt')
        else:
                print 'Saving to topwords.txt'
                #'/home1/c/cis530/hw3/xml_data'
                top_words = extract_top_words('/home1/c/cis530/hw3/xml_data')
                save_object(top_words, cwd + '/topwords.txt')
	#print top_words
	
	if os.path.isfile(cwd + '/similarity.txt'):
                print 'Loading from similarity.txt'
                sim_mat = load_object(cwd + '/similarity.txt')
        else:
                print 'Saving to similarity.txt'
                #'/home1/c/cis530/hw3/xml_data'
                sim_mat = extract_similarity(top_words)
                save_object(sim_mat, cwd + '/similarity.txt')
	#print sim_mat

	if os.path.isfile(cwd + '/dependency.txt'):
                print 'Loading from dependency.txt'
                top_dep = load_object(cwd + '/dependency.txt')
        else:
                print 'Saving to dependency.txt'
                #'/home1/c/cis530/hw3/xml_data'
                top_dep = extract_top_dependencies('/home1/c/cis530/hw3/xml_data')
                save_object(top_dep, cwd + '/dependency.txt')
        """
	#print top_dep
	#xml_directory = "/home1/c/cis530/hw3/xml_data"
	#process_corpus( xml_directory, top_words, sim_mat, top_dep, prod_rules )
	#xml_directory = "test_data"
	#process_corpus( xml_directory, top_words, sim_mat, top_dep, prod_rules )
	#generate_liblinear_files()
	#print run_classifier("Computers_train_1.txt","Computers_test_1.txt")
	#generate_results()
	#generate_classify_results()
