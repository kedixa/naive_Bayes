# Makefile for naive_bayes

naive_bayes.out: main.cpp naive_bayes.h naive_bayes.cpp
	g++ -std=c++11 -O2 main.cpp naive_bayes.cpp -o naive_bayes.out

.PHONY: clean

clean:
	rm naive_bayes.out
