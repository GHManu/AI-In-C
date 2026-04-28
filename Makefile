all:
	gcc main.c logistic_regression.c -o main -lm

clean:
	rm -f main