#include "stringList.h"
#include <stdio.h>
#include <iostream>

//Constructor
stringList::stringList()
{
	//initilize everything to empty
	m_size = 0;
	m_listStart = NULL;
	m_listEnd = NULL;
}

//retrieve list size
int stringList::getSize()
{
	return m_size;
}

//retrieve pointer to list start
stringListPiece* stringList::getListStart()
{
	return m_listStart;
}

//retrieve point to list end
stringListPiece* stringList::getListEnd()
{
	return m_listEnd;
}

//add string to end of the list
//initilize list starting point if this is the first entry
//increment list size count
int stringList::add(stringListPiece* newString)
{

	if(m_listStart==NULL)
		m_listStart=newString;
	else
		m_listEnd->nextString = newString;

	m_listEnd = newString;
	m_listEnd->nextString = NULL;

	return m_size++;
}

bool stringList::remove(int index)
{
	if(index >= m_size ) return false;

	stringListPiece* m_remove;
	stringListPiece* currentPiece;

	if(m_size == 1) {
		m_remove = m_listStart;
		m_listStart = NULL;
		m_listEnd = NULL;
	} else {

		if(index == 0) {
			m_remove = m_listStart;
			m_listStart = m_remove->nextString;
		} else if(index == m_size - 1) {
			m_remove = m_listEnd;
			currentPiece = m_listStart;
			for(int i=1; i<m_size-1; i++)
			{
				currentPiece = currentPiece->nextString;
			}
			currentPiece->nextString = NULL;
			m_listEnd = currentPiece;
		} else {
			currentPiece = m_listStart;
			for(int i=1; i<=index-1; i++)
			{
				currentPiece = currentPiece->nextString;
			}
			m_remove = currentPiece->nextString;
			currentPiece->nextString = m_remove->nextString;
		}
	}

	delete m_remove;
	m_size -= 1;
	return true;
}

//print out all the Decuda Instructions in the list
void stringList::printStringList()
{
	stringListPiece* currentPiece = m_listStart;

	for(int i=0; (i<m_size)&&(currentPiece!=NULL); i++)
	{
		std::cout << currentPiece->stringText << " ";
		currentPiece = currentPiece->nextString;
	}
}
