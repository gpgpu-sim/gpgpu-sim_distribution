TARGET := post-traces-processing

$(TARGET): post-traces-processing.cpp
	g++ -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean: 
	rm -f $(TARGET) *.o

