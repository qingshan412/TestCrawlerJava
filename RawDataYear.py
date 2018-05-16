#!/usr/bin/env python
# -*- coding: utf-8 -*-
'Download References'
__author__='qingshan412'

import itertools, copy
import numpy as np
import urllib, urllib2, cookielib
import os,sys#,types
import re
from bs4 import BeautifulSoup


### Function to get search result url's appendix
def SearchKeyWords(*args):
	for i in range(len(args)):
		if i!=0:
			SearchWordsi += ('+'+args[i])
		else:
			SearchWordsi = args[i]
	return SearchWordsi

### Function to find paths iteratively
def FindPath(ThisNode,Paths,M,T,TmpPath,NodeData):
	cite_nodes=NodeData[ThisNode][3]
	for node_i in cite_nodes:
		#print(node_i)
		if node_i in T:
			#t_p_t=TmpPath
			t_p=copy.deepcopy(TmpPath)
			t_p.append(node_i)
			Paths.append(t_p)
			#print('t:')
			#print(t_p)
			#print(TmpPath)
		elif node_i in M:
			#t_p=TmpPath
			t_p=copy.deepcopy(TmpPath)
			#print('m:')
			#print(t_p)
			t_p.append(node_i)
			#print(t_p)
			FindPath(node_i,Paths,M,T,t_p,NodeData)
	
	
	
### Get KW in
KeyWord = raw_input('give keywords: ')
KeyWordF = KeyWord.lower().split('+')
#raw_input(KeyWordF)
KeyWord = KeyWord.strip()
#raw_input(KeyWord)
#KeyWordF = KeyWord.lower().split()
#KeyWord = KeyWord.split()
#KeyWordF = KeyWord[0]
#print(KeyWordF)

### set dowload path and txt path
PaperPath = 'Papers' + os.path.sep + KeyWord + os.path.sep

#for i in range(len(KeyWord)):
#	if i != 0:
#		PaperPath = PaperPath + '+' + KeyWord[i]
#	else:
#		PaperPath = PaperPath + KeyWord[i]
#PaperPath = PaperPath + os.path.sep 

if not os.path.exists(PaperPath):
	os.makedirs(PaperPath)

PaperPathPdf = PaperPath + 'Pdf' + os.path.sep 
if not os.path.exists(PaperPathPdf):
	os.makedirs(PaperPathPdf)


PaperPathTxt = PaperPath + 'Txt' + os.path.sep 
if not os.path.exists(PaperPathTxt):
	os.makedirs(PaperPathTxt)

###manually homepage path
PaperPathHP = PaperPath + 'HomePage' + os.path.sep
PaperPathSy = PaperPath + 'survey' + os.path.sep
if not os.path.exists(PaperPathHP) or not os.path.exists(PaperPathSy):
	raw_input('wrong keyword!')
	sys.exit(0)

print(PaperPathPdf)
print(PaperPathTxt)
raw_input(PaperPathHP)

### KW after permutation
#KeyWords = []
#for i in range(1,len(KeyWord)+1):
#	Keyword = itertools.combinations(KeyWord,i)#permutations(KeyWord,i)   it should be permutation
#	KeyWords += list(Keyword)


### prepare for detecting where to download pdfs
print 'begin to search!'
PatternInnerLink = re.compile("\/stamp\/stamp\.jsp\?tp=&arnumber=\d+")
#re.compile("http:\/\/ieeexplore\.ieee\.org\/xpl\/articleDetails\.jsp\?arnumber=\d+")
PatternPaperNum = re.compile('\d+')
PatternPDF = re.compile('http:\/\/ieeexplore\.ieee\.org.*\.pdf[^"]+')
#PaperNum = []
PDFLinks = []
Num_Title_Count_Ref_Cit_Year_Link_Pos={}
#OuterLink='http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber='

### get all links and titles in manually download htmls
for filename in os.listdir(PaperPathHP):
	#print filename
	content_homepage = open(PaperPathHP+filename).read()
	soup=BeautifulSoup(content_homepage)
	#print(soup.prettify())
	#raw_input('...')
	#print(soup.prettify())
	tag_div=soup.find_all('div', "pure-u-22-24")
	for tag in tag_div:
		InnerLink=tag.find('a',href=PatternInnerLink)
		if InnerLink:
			InnerLink=InnerLink['href']
			print(InnerLink)
		
			title=tag.find('h2')
			title=title.get_text()
			title=title.strip()
			print(title)
			
			year=tag.div.div.span#find('span',"ng-binding ng-scope")
			year=year.get_text()
			print(year)
			year=re.search(PatternPaperNum,year).group()
			print(year)
			
			num=re.search(PatternPaperNum,InnerLink).group()
			print(num)
			
			if not Num_Title_Count_Ref_Cit_Year_Link_Pos.has_key(num) and num[0]!='6':
				Num_Title_Count_Ref_Cit_Year_Link_Pos[num]=[title,{},[],[],year,InnerLink,[]]
	#print(Num_Title_Count_Ref_Cit_Year_Link_Pos)
### no link in title
#Num_Title_Count_Ref_Cit_Year_Link_Pos['1414318']=['Intellectual Properties Protection in VLSI Designs: Theory and Practice ',{},[],[]]
#print(Num_Title_Count_Ref_Cit_Year_Link_Pos.has_key('1414318'))
#print(Num_Title_Count_Ref_Cit_Year_Link_Pos.has_key('1297233'))



raw_input('pause...')




for num in Num_Title_Count_Ref_Cit_Year_Link_Pos.keys():
	if not os.path.isfile(PaperPathPdf + num + '.pdf'):
		### get outerlink for pdf
		cj = cookielib.CookieJar()			### cookie using
		opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
		urllib2.install_opener(opener)
		#req = urllib2.Request('http://ieeexplore.ieee.org' + innerlink,None,req_header)	### manual cookie
		resp = urllib2.urlopen(Num_Title_Count_Ref_Cit_Year_Link_Pos[num][5])
		content = resp.read()
					
		### get download link for pdf
		PDFlink = re.search(PatternPDF, content)#.group()
		if PDFlink:
			PDFlink = PDFlink.group()
			PDFlink = PDFlink + '&tag=1'
			print PDFlink
			PDFLinks.append(PDFlink) 
			
			resp = urllib2.urlopen(PDFlink)
			content = resp.read()
			
			### save pdfs
			f = open(PaperPathPdf + num + '.pdf', 'w')
			f.write(content)
			f.close()
						
		else:
			raw_input('cookie is out of date...')

### transform pdfs into txt
for filename in os.listdir(PaperPathPdf):
	print filename
	appendix = filename.split('.')
	if not os.path.exists(PaperPathTxt+appendix[0]+'.txt'):
		if appendix[-1]=='pdf':
			os.environ['PaperPathPdf']=str(PaperPathPdf+appendix[0]+'.pdf')# equals to PaperPathPdf+filename
			os.environ['PaperPathTxt']=str(PaperPathTxt+appendix[0]+'.txt')
			os.system('pdf2txt.py -o $PaperPathTxt $PaperPathPdf')
			
### count keywords + reference&citation
### prepare for delete extra chars
PatternNonWord=re.compile('\W')
for filename in os.listdir(PaperPathTxt):
	appendix = filename.split('.')
	text = open(PaperPathTxt+filename).read().lower()
	### count keywords
	PosKw=[]
	CountKw={}
	#KeyWordF=[KeyWord.lower()]
	for keyword in KeyWordF:
		kw_test=PatternNonWord.sub('',keyword)
		kw_len=len(kw_test)
		kw=''
		for i in range(kw_len):
			if i<(kw_len-1):
				kw=kw+kw_test[i]+'\W*'
			else:
				kw=kw+kw_test[i]
		#CountKw[keyword]=0
		kw_num=re.findall(kw,text)
		if kw_num:
			CountKw[keyword]=len(kw_num)
			pos=re.finditer(kw,text)
			for item in pos:
				Pos_t=item.span()
				Pos_t=Pos_t[0]
				PosKw.append(Pos_t)
	Num_Title_Count_Ref_Cit_Year_Link_Pos[appendix[0]][1]=CountKw
	Num_Title_Count_Ref_Cit_Year_Link_Pos[appendix[0]][6]=PosKw
	#Num_Title_Count_Ref_Cit_Year_Link_Pos[appendix[0]].append(CountKw)
	
	### reference&citation
	RefList=[]
	for (k, v) in Num_Title_Count_Ref_Cit_Year_Link_Pos.items():
		if k!=appendix[0]:
			title_test=PatternNonWord.sub('',v[0].lower())
			title_len=len(title_test)
			title=''
			for i in range(title_len):
				if i<(title_len-1):
					title=title+title_test[i]+'\W*'
				else:
					title=title+title_test[i]
			if re.search(title,text):
				RefList.append(k)
				Num_Title_Count_Ref_Cit_Year_Link_Pos[k][3].append(appendix[0])
	Num_Title_Count_Ref_Cit_Year_Link_Pos[appendix[0]][2]=RefList


### feature data
f = open(PaperPath + KeyWord + '+tmp.txt', 'w')
for (k,v) in Num_Title_Count_Ref_Cit_Year_Link_Pos.items():
	result=0
	text = open(PaperPathSy+'survey_refer_number.txt','r')
	for eachline in text:
		eachline=PatternNonWord.sub('',eachline)
   		if str(k)==str(eachline):
			result=1
			#print(result)
			break
		else:
			pass
			#print(eachline)
	
	para1=0
	if v[1]:
		for (k1,v1) in v[1].items():
			para1=para1+v1
	#raw_input(para1)		### keywords count
	para2=len(v[2])			### Ref count
	para3=len(v[3])			### Citation count
	para4=v[4]				### year
	
	if v[6]:
		para5_t=np.array(v[6])	### keywords' postions
		#raw_input(para5_t)
		para5=para5_t.mean()
		para6=para5_t.var()
		para7=para5_t.min()
		para8=para5_t.max()
	else:
		para5=0
		para6=0
		para7=0
		para8=0
	
	para9_t=[]				### references' published year
	if v[2]:
		for item in v[2]:
			para9_t.append(int(Num_Title_Count_Ref_Cit_Year_Link_Pos[item][4]))
		para9_t=np.array(para9_t)
		#raw_input(para9_t)
		para9=para9_t.mean()
		para10=para9_t.var()
		para11=para9_t.min()
		para12=para9_t.max()
	else:
		para9=0
		para10=0
		para11=0
		para12=0
	
	LineData=str(para1)+'\t'+str(para2)+'\t'+str(para3)+'\t'+str(para4)+'\t'+str(para5)+'\t'+str(para6)+'\t'+str(para7)+'\t'+str(para8)+'\t'+str(para9)+'\t'+str(para10)+'\t'+str(para11)+'\t'+str(para12)+'\t'+str(result)+'\n'
	f.write(LineData)
	#print(v)
f.close()

### Path

raw_input('pause...')