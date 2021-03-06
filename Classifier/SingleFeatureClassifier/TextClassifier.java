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
    public static final int NUMBER_GRAM = 3;


    public Properties props;
    public StanfordCoreNLP pipeline;
    public Charset charset;
    public HashSet<String> random_list;
    public Map<String, Integer> global_dict;
    public int fn;	//TAG number
    public ArrayList<Integer> ftlist;
    public String wFileName;
    public String[] grams;
    public String entire_gram;

    public LexicalizedParser lp;
    
    public void ParseFileGram(BufferedReader reader, File file_to_write, int is_phishing) {
      String line;
      try(BufferedWriter svmwriter = new BufferedWriter(new FileWriter(file_to_write.getName(), true))){
        ftlist.clear();
        svmwriter.write(is_phishing +" ");
        while(((line=reader.readLine())!=null) && (line.length()>0)){
	  //counter++;
          Annotation document = new Annotation(line);
          pipeline.annotate(document);
          List<CoreMap> sentences = document.get(SentencesAnnotation.class);
          for(CoreMap sentence : sentences){
                int counter = 0;
                for(CoreLabel token: sentence.get(TokensAnnotation.class)){
                  String word = token.get(TextAnnotation.class);
                  //check if the word in the global dictionary
                  word = word.toLowerCase();
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
                  if(!global_dict.containsKey(entire_gram)){
                    global_dict.put(entire_gram, global_dict.size()+1);
                  }
		  //obtain the integer correpsondes to the gram
		  //and add the integer to feature list
                  fn = global_dict.get(entire_gram);
                  ftlist.add(fn);

                }

            }
        }
        HashSet<Integer> hs = new HashSet<Integer>();
        hs.addAll(ftlist);
        ftlist.clear();
        ftlist.addAll(hs);
        Collections.sort(ftlist);
        for(int j=0; j<ftlist.size(); j++){
          svmwriter.write(ftlist.get(j) + ":1 ");
          //System.out.println(ftlist.get(j) + "is added to svmwriter");
        }
        svmwriter.write("\t#" + wFileName);
        svmwriter.newLine();
      } catch(IOException x){
        System.err.format("IOException: %s%n", x);
      }
    }


    //both global_dict and ftlist are global
    public void AssembleFeatureFromTree(Tree parse) {
      String feature=parse.value() + "->";
      Tree[] sub_tree = parse.children();
      for(Tree t: sub_tree) {
        if (!t.isPreTerminal()) {
          AssembleFeatureFromTree(t);
        }
        feature+=t.value() + "+";
      }
      
      if(!global_dict.containsKey(feature)){
        global_dict.put(feature, global_dict.size()+1);
      }
      fn = global_dict.get(feature);
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

                AssembleFeatureFromTree(parse);

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

    public void DoTheWork(String[] args){
	String train_file_name = "essay_ab_svmtraing35.dat";
	String classify_file_name = "essay_ab_svmclassifyg35.dat";
        LoadRandomList();
	props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        pipeline = new StanfordCoreNLP(props);
	charset = Charset.forName("UTF-8");
	//charset.onMalformedInput(CodingErrorAction.IGNORE);
	//String rpath = "/homes/jind/TestData/";
	//String rpath="/homes/jind/newtestdata/";
	String rpath = "/homes/jind/OtherDataSet/formattedessay/";
	File folder = new File(rpath);
	File[] listOfFiles = folder.listFiles();
        //int counter = 1;
	global_dict = new HashMap<String, Integer>();
	fn = 0;	//TAG number
        grams = new String[NUMBER_GRAM];
        entire_gram = "";
	//create a feature list with 2500 features, this list is to maintain current sentence feature
	ftlist = new ArrayList<Integer>(2500);
	//this int array is used to record training data frequency of each tag type
        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        lp = LexicalizedParser.loadModel(parserModel);


	//create svm training data file to store the converted data in svm required form
	File svmtfile = new File(train_file_name);
	try{
	    if(svmtfile.exists()){
		svmtfile.delete();
	    }
	    svmtfile.createNewFile();
	} catch(Exception x){
	    x.printStackTrace();
	}
	//create svm classify data file
	File svmcfile = new File(classify_file_name);
	try{
	    if(svmcfile.exists()){
		svmcfile.delete();
	    }
	    svmcfile.createNewFile();
	} catch(Exception x){
	    x.printStackTrace();
	}
        //1234,12345,123456
        //Random random_gen = new Random(123456);

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
		//System.out.println(wFileName);
		Path srcfilepath = Paths.get(rpath+wFileName);
		try(BufferedReader reader = Files.newBufferedReader(srcfilepath, charset)){
		    //write to the training data
                    //if ((random_gen.nextInt() % 3) !=0) {
		    int ind = Integer.parseInt(wFileName);
		    //This is a three fold approach
		    //if((ind<67) || ((ind>499) && (ind<567))){
		    //if(((ind>66) && (ind<134)) || ((ind>566)&& (ind<634))){
		    //if(((ind>133) && (ind<201)) || (ind>633)){

                    //5-fold for within topic essay test
                    //if((ind<80) ||((ind>499) && (ind<580))){
		    //if(((ind<60)||((ind>79)&&(ind<100))
		    //    ||(((ind>499)&&(ind<560))||(ind>579)&&(ind<600)))){
		    //if(((ind<40)||((ind>59)&&(ind<100)))
		    //    ||(((ind>499)&&(ind<540))||((ind>559)&&(ind<600)))){
		    //if(((ind<20)||((ind>39)&&(ind<100)))
		    //    ||(((ind>499)&&(ind<520))||((ind>539)&&(ind<600)))){
		    if(((ind>19)&&(ind<100)) || ((ind>519)&&(ind<600))){
                       //this is the training data
		       ParseFileGram(reader, svmtfile, is_phishing);
                       //ParseFileTree(reader, svmtfile, is_phishing);
                    } else {
                       //this is the testing data

		       //if(((ind>79) && (ind<100)) || ((ind>579)&&(ind<600))){
		       //if(((ind>59)&&(ind<80)) || ((ind>559)&&(ind<580))){
		       //if(((ind>39)&&(ind<60))||((ind>539)&&(ind<560))){
		       //if(((ind>19)&&(ind<40))||((ind>519)&&(ind<540))){
		       if((ind<20)||((ind>499)&&(ind<520))){
                        ParseFileGram(reader, svmcfile, is_phishing);
			//ParseFileTree(reader, svmcfile, is_phishing);
		       }
                    }
		} catch(IOException x){
		    System.err.format("IOException: %s%n", x);
		}
	    }
            System.out.println(i);
	}

    }
    
    public static void main(String[] args) {
      TextClassifier textClassifier = new TextClassifier();
      textClassifier.DoTheWork(args);
    } 
}    
