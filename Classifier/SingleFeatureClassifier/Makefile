STANFORD_CORENLP_PATH=$(HOME)/NLPTools/stanford-corenlp-full-2015-01-29
STANFORD_PARSER_PATH=$(HOME)/NLPTools/stanford-parser-full-2015-01-30

all: TextClassifier.class

TextClassifier.class : TextClassifier.java
	javac -cp $(STANFORD_CORENLP_PATH)/stanford-corenlp-3.5.1.jar:$(STANFORD_PARSER_PATH)/stanford-parser.jar:./ $^

run:
	java -cp $(STANFORD_CORENLP_PATH)/*:$(STANFORD_PARSER_PATH)/*:. TextClassifier

clean: FORCE
	rm *.class

FORCE:
