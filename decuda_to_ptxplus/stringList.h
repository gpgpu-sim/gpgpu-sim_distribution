struct stringListPiece
{
	const char* stringText;
	
	stringListPiece* nextString;
};


class stringList
{

private:
	//List size
	int m_size;

	//Start is the first entry, end is the last entry
	stringListPiece* m_listStart;
	stringListPiece* m_listEnd;
public:
	//constructor
	stringList();

	//accessors
	int getSize();
	stringListPiece* getListStart();
	stringListPiece* getListEnd();

	//mutator
	int add(stringListPiece* newString); //add String to list
	bool remove(int index); //remove string at index

	//print representation
	void printStringList();
};
