import java.nio.charset.Charset;
import java.io.*;
import java.nio.file.*;
import java.util.*;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.io.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;


public class TextClassifier{
    //set up n-gram
    public static final int NUMBER_GRAM = 2;
    public static final int PARSE_BY_GRAM = 0;
    public static final int PARSE_BY_TREE = 1;
    //public static final int PARSE_METHOD = PARSE_BY_GRAM;
    public static final int PARSE_METHOD = PARSE_BY_TREE;

    public Properties props;
    public StanfordCoreNLP pipeline;
    public Charset charset;
    public HashSet<String> random_list;
    public int total_doc;
    // map feature -> feature_index;
    public Map<String, Integer> global_feature_index;

    // map feature -> number of docs that has the feature
    public Map<String, Integer> global_idf_index;

    // map doc index -> length of the doc;
    public Map<Integer, Integer> global_doc_length;
    public String wFileName;
    public String[] grams;
    public String entire_gram;
    public LexicalizedParser lp;

/*
    public void AssembleFeatureFromTree(Tree parse, boolean isFirstCall) {
      String feature=parse.value() + "->";
      Tree[] sub_tree = parse.children();
      for(Tree t: sub_tree) {
        if (!t.isPreTerminal()) {
          AssembleFeatureFromTree(t);
        }
        feature+=t.value() + "+";
      }
      if (isFirstCall) {
        if(!global_dict.containsKey(feature)){
          global_dict.put(feature, global_dict.size()+1);
        }
      } else if (!global_dict.containKey(feature)) {
        System.out.println("global should contain key: "+ feature);
      }
      int fn = global_dict.get(feature);
      //System.out.println(feature + " : " + fn);
      ftlist.add(fn);
    }

    public void ParseFileTree(BufferedReader reader, File file_to_write, int is_phishing) {
      String line;
      try(BufferedWriter svmwriter = new BufferedWriter(new FileWriter(file_to_write.getName(), true))){
        ftlist.clear();
        svmwriter.write(is_phishing+" ");
        while((line=reader.readLine())!=null){
          if (line.length() == 0) {
            continue;
          }
	  //counter++;
          Annotation document = new Annotation(line);
          pipeline.annotate(document);
          List<CoreMap> sentences = document.get(SentencesAnnotation.class);
          for(CoreMap sentence : sentences){
                //svmwriter.write(GetType(type,typecounter) + " qid:" + (i+1) + " ");
                ArrayList<String> token_vec = new ArrayList<String>();
                for(CoreLabel token: sentence.get(TokensAnnotation.class)){
                  String word = token.get(TextAnnotation.class);
                  //check if the word in the global dictionary
                  word = word.toLowerCase();
                  token_vec.add(word);
                }

                String[] sent = new String[token_vec.size()];
                sent = token_vec.toArray(sent);
                TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
                List<CoreLabel> rawWords = Sentence.toCoreLabelList(sent);
                Tree parse = lp.apply(rawWords);
                //parse.pennPrint();

                AssembleFeatureFromTree(parse, false);

          }
        }

        HashSet<Integer> hs = new HashSet<Integer>();
        hs.addAll(ftlist);
        ftlist.clear();
        ftlist.addAll(hs);
        Collections.sort(ftlist);
        for(int j=0; j<ftlist.size(); j++){
          svmwriter.write(ftlist.get(j) + ":1 ");
	}
	svmwriter.write("\t#" + wFileName);
	svmwriter.newLine();
      } catch(IOException x){
        System.err.format("IOException: %s%n", x);
      }
    }
*/

    public void LoadRandomList(){
      String line;
      random_list = new HashSet<String>();
      Path random_path = Paths.get("./essaylegitlist.txt");
      try(BufferedReader reader = Files.newBufferedReader(random_path)){
          while((line=reader.readLine())!=null){
	      random_list.add(line);
	  }
      } catch(IOException x){
        System.err.format("IOException: %s%n", x);
      }
    }

    public void AssembleFeatureFromTree(Tree parse, Map<String, Integer> feature_freq_dict){
	String feature = parse.value() + "->";
	Tree[] sub_tree = parse.children();
	for(Tree t: sub_tree){
	    if(!t.isPreTerminal()){
		AssembleFeatureFromTree(t, feature_freq_dict);
            }
	    feature+=t.value() + "+";
	}
	if(feature_freq_dict.containsKey(feature)){
	    feature_freq_dict.put(feature, feature_freq_dict.get(feature)+1);
	} else {
	    feature_freq_dict.put(feature, 1);
	}
	    
    }

    public void parseSentenceByTree(
        CoreMap sentence,
	Map<String, Integer> feature_freq_dict) {
      // function stub for parse by tree
        ArrayList<String> token_vec = new ArrayList<String>();
	for(CoreLabel token: sentence.get(TokensAnnotation.class)){
	    String word = token.get(TextAnnotation.class);
	    word = word.toLowerCase();
	    token_vec.add(word);
	}

	String[] sent = new String[token_vec.size()];
	sent = token_vec.toArray(sent);
	TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
	List<CoreLabel> rawWords = Sentence.toCoreLabelList(sent);
	Tree parse = lp.apply(rawWords);

	AssembleFeatureFromTree(parse, feature_freq_dict);
    }

    public void parseSentenceByGram(
        CoreMap sentence,
        Map<String, Integer> feature_freq_dict) {
            int counter = 0;
            for(CoreLabel token: sentence.get(TokensAnnotation.class)){
                String word = token.get(TextAnnotation.class);
                //check if the word in the global dictionary
                word = word.toLowerCase();
                //for test purpose
                //System.out.println(word + " is added");
                counter = counter + 1;
                if (counter < NUMBER_GRAM) {
                grams[counter] = word;
                continue;
            } else {
                entire_gram = "";
                for(int j=0;j<NUMBER_GRAM-1;j++) {
                    grams[j] = grams[j+1];
                    entire_gram += grams[j];
                }
                grams[NUMBER_GRAM-1] = word;
                entire_gram += word;
            }
            //Put the gram into dictionary,
            //The gram corresponds to an integer in the dictionary
            if(feature_freq_dict.containsKey(entire_gram)){
                feature_freq_dict.put(entire_gram, feature_freq_dict.get(entire_gram)+1);
            } else {
                feature_freq_dict.put(entire_gram, 1);
            }
        }
    }

    public Map<String, Integer> FileParser(
        int i, // File) 
	BufferedReader reader, // fileReader,
        boolean isBuildingOutput,  // Is Building output
        int parseMethod,  // parse By Gram or Tree,
        String wFileName, // write File Name
	File file_to_write,
        int tag // the class tag of the file
        ) throws Exception {
      Map<String, Integer> feature_freq_dict = new HashMap<String, Integer>();
      String line;
      if (reader == null) {
	      System.out.println("ERR!!Nothing to be read!!");
	      System.exit(-1);
      }
      while(((line = reader.readLine()) != null) && (line.length() > 0 )) {
        Annotation document = new Annotation(line);
	pipeline.annotate(document);
	List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	for (CoreMap sentence : sentences) {
          if (parseMethod == PARSE_BY_GRAM) {
            parseSentenceByGram(sentence, feature_freq_dict);
          } else {
            parseSentenceByTree(sentence, feature_freq_dict);
          }
        }
      }
      // now we have the entire feature_freq_dict for the document
      if (isBuildingOutput) {
        // Write output to file here
	BufferedWriter svmwriter = new BufferedWriter(new FileWriter(file_to_write.getName(), true));
	svmwriter.write(tag + " ");
	int local_doc_length = 0;
	TreeMap<Integer, Double> local_feature_map = new TreeMap<>();
        for(String feature: feature_freq_dict.keySet()) {
          // feature number;
          int fn = global_feature_index.get(feature);
          // local feature frequncy;
          double local_freq = (double)feature_freq_dict.get(feature);
          // number of docs with this feature;
          double doc_freq = (double)global_idf_index.get(feature);
	  double tfidf = (local_freq/global_doc_length.get(i)) * Math.log(total_doc/doc_freq);
	  local_feature_map.put(fn, tfidf);
	}
	for(Integer fn : local_feature_map.keySet()) {
          svmwriter.write(fn + ":" + local_feature_map.get(fn) + " ");
        }
	svmwriter.write("\t # "+ wFileName + "\n");
	svmwriter.close();
      }
      return feature_freq_dict;
    }

    public void DoTheWork(String[] args){
	String train_file_name = "essay_bf_svmtraint5.dat";
	String classify_file_name = "essay_bf_svmclassifyt5.dat";
        LoadRandomList();
	props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        pipeline = new StanfordCoreNLP(props);
	charset = Charset.forName("UTF-8");
	//charset.onMalformedInput(CodingErrorAction.IGNORE);
	//String rpath = "/homes/jind/TestData/";
	//String rpath="/homes/jind/newtestdata/";
	String rpath = "/homes/jind/OtherDataSet/formattedessay/";
	//String rpath = "/homes/jind/OtherDataSet/testdataset/";
	File folder = new File(rpath);
	File[] listOfFiles = folder.listFiles();
        //int counter = 1;
	global_feature_index = new HashMap<String, Integer>();
	global_idf_index = new HashMap<String, Integer>();
	global_doc_length = new HashMap<Integer, Integer>();
        grams = new String[NUMBER_GRAM];
        entire_gram = "";
        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        lp = LexicalizedParser.loadModel(parserModel);


	//create svm training data file to store the converted data in svm required form
	File svm_t_file = new File(train_file_name);
	try{
	    if(svm_t_file.exists()){
		svm_t_file.delete();
	    }
	    svm_t_file.createNewFile();
	} catch(Exception x){
	    x.printStackTrace();
	}
	//create svm classify data file
	File svm_c_file = new File(classify_file_name);
	try{
	    if(svm_c_file.exists()){
		svm_c_file.delete();
	    }
	    svm_c_file.createNewFile();
	} catch(Exception x){
	    x.printStackTrace();
	}
        
	total_doc = listOfFiles.length;
	// The first round of parseing all files to build the tf-idf index
        for(int i=0; i<listOfFiles.length; i++){
	  wFileName = listOfFiles[i].getName();
	  Path srcfilepath = Paths.get(rpath+wFileName);
	  try {
            BufferedReader reader = Files.newBufferedReader(srcfilepath, charset);
            int ind = Integer.parseInt(wFileName);
	    Map<String, Integer> feature_freq_dict;
	    feature_freq_dict = FileParser(i, // fileIndex,
	               reader, // fileREader,
		       false,  // Is Building output
		       PARSE_METHOD,  // parse method,
		       wFileName, // write File Name
		       null,
                       0);
            // Build global feature index here 
	    int local_doc_length = 0;
            for(String feature : feature_freq_dict.keySet()) {
              local_doc_length += feature_freq_dict.get(feature);
              if (!global_feature_index.containsKey(feature)) {
                global_feature_index.put(feature, global_feature_index.size() + 1);
              }
              if (!global_idf_index.containsKey(feature)) {
                global_idf_index.put(feature, 1);
              } else {
                global_idf_index.put(feature, global_idf_index.get(feature) + 1);
              }
            }
	    global_doc_length.put(i, local_doc_length);
	    reader.close();
	  } catch(Exception x) {
             System.err.format("Exception: %s%n", x);
	     x.printStackTrace();
	  }
	}
        // The second parse of all files generates the svm output
	for(int i=0; i<listOfFiles.length; i++){
	    if(listOfFiles[i].isFile()){
                
		wFileName = listOfFiles[i].getName();
                int is_phishing = -1; //-1 means  phishing
                if (random_list.contains(wFileName)) {
                  System.out.println(wFileName + "is NOT phishing. ???");
                  is_phishing = 1;
                } else {
		   System.out.println(wFileName + "is phishing. ???");
		}
		Path srcfilepath = Paths.get(rpath+wFileName);
	 	try(BufferedReader reader = Files.newBufferedReader(srcfilepath, charset)){
		    //write to the training data
		    int ind = Integer.parseInt(wFileName);
                    //5-fold for within topic essay test
                    //if(((ind>199)&&(ind<280)) ||((ind>699) && (ind<780))){
		    //if((((ind>199)&&(ind<260))||((ind>279)&&(ind<300))
		    //    ||(((ind>699)&&(ind<760))||(ind>779)&&(ind<800)))){
		    //if((((ind>199)&&(ind<240))||((ind>259)&&(ind<300)))
		    //    ||(((ind>699)&&(ind<740))||((ind>759)&&(ind<800)))){
		    //if((((ind>199)&&(ind<220))||((ind>239)&&(ind<300)))
		    //    ||(((ind>699)&&(ind<720))||((ind>739)&&(ind<800)))){
		    if(((ind>219)&&(ind<300)) || ((ind>719)&&(ind<800))){
                        //this is the training data
	               FileParser(i, // fileIndex,
	                   reader, // fileREader,
		           true,  // Is Building output
		           PARSE_METHOD,  // parse_method
		           wFileName, // write File Name
			   svm_t_file,
                           is_phishing);
                    } else {
                       //this is the testing data

		       //if(((ind>279) && (ind<300)) || ((ind>779)&&(ind<800))){
		       //if(((ind>259)&&(ind<280)) || ((ind>759)&&(ind<780))){
		       //if(((ind>239)&&(ind<260))||((ind>739)&&(ind<760))){
		       //if(((ind>219)&&(ind<240))||((ind>719)&&(ind<740))){
		       if(((ind>199)&&(ind<220))||((ind>699)&&(ind<720))){
	                 FileParser(i, // fileIndex,
	                     reader, // fileREader,
		             true,  // Is Building output
		             PARSE_METHOD,  // parse_method,
		             wFileName, // write File Name
			     svm_c_file,
                             is_phishing);
		       }
                    }
                  reader.close();
		} catch(Exception x){
		    System.err.format("IOException: %s%n", x);
		    x.printStackTrace();
		}
		//}
	    }
            System.out.println(i);
	}
    }
    
    public static void main(String[] args) {
      TextClassifier textClassifier = new TextClassifier();
      textClassifier.DoTheWork(args);
    } 
}    
