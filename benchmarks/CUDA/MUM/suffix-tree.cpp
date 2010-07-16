#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <sys/time.h>
#include <list>
#include <map>
#include <vector>
#include <queue>

#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <errno.h>
#include <assert.h>

#define ulong4 uint32_t
#include "mummergpu.h"

#define MPOOL 1

#include "PoolMalloc.hh"


#include <string.h>

using namespace std;

// Enable verification/debug options
#define VERIFY  0
#define VERBOSE 0
const bool DEBUG = 0;

// Setting for linear time alg
bool FORCEROOT = false;
bool DOJUMP = true;
bool DOINTERNALSKIP = true;
bool DOPHASETRICK = true;

// Statistics
int skippedbases = 0;
int skippedextensions = 0;

char substrbuffer[1024];
const char * substr(const char * str, int start, int len)
{
  if (len > 1024) { len = 1024; }
  strncpy(substrbuffer, str+start, len);
  substrbuffer[len] = '\0';

  return substrbuffer;
}


// Helper to convert from ascii to single byte
unsigned char b2i(char base)
{
  switch (base)
  {
    case 'A' : return 0;
    case 'C' : return 1;
    case 'G' : return 2;
    case 'T' : return 3;
    case '$' : return 4;

    default: 
      cerr << "Unknown base: " << base << endl;
      return b2i('A');
  };
}

#include <sys/time.h>
#include <string>

class EventTime_t
{
public:
  /// Constructor, starts the stopwatch
  EventTime_t()
  {
    start();
    memset(&m_end, 0, sizeof(struct timeval));
  }


  /// Explicitly restart the stopwatch
  void start()
  {
    gettimeofday(&m_start, NULL);
  }


  /// Explicitly stop the stopwatch
  void stop()
  {
    gettimeofday(&m_end, NULL);
  }


  /// Return the duration in seconds
  double duration()
  {
    if ((m_end.tv_sec == 0) && (m_end.tv_usec == 0)) { stop(); }
    return ((m_end.tv_sec - m_start.tv_sec)*1000000.0 + (m_end.tv_usec - m_start.tv_usec)) / 1e6;
  }


  /** \brief Pretty-print the duration in seconds.
   ** If stop() has not already been called, uses the current time as the end
   ** time.
   ** \param format Controls if time should be enclosed in [ ] 
   ** \param precision Controls number of digits past decimal pt
   **/
  std::string str(bool format = true, 
                  int precision=2)
  {
    double r = duration();

    char buffer[1024];
    sprintf(buffer, "%0.*f", precision, r);

    if (format)
    {
      string s("[");
      s += buffer;
      s += "s]";
      return s;
    }

    return buffer;
  }


private:
  /// Start time
  struct timeval m_start;

  /// End time
  struct timeval m_end;
};


// A node in the suffix tree
class SuffixNode
{
public:
  static int s_nodecount;

#ifdef MPOOL
  void *operator new( size_t num_bytes, PoolMalloc_t *mem)
  {
    return mem->pmalloc(num_bytes);
  }
#endif

  SuffixNode(int s, int e, int leafid,
             SuffixNode * p, SuffixNode * x)
    : m_start(s), m_end(e), 
      m_nodeid(++s_nodecount),
      m_leafid(leafid),
      m_parent(p), 
	  m_suffix(x)
  {
    for (int i = 0; i < basecount; i++)
    { m_children[i] = NULL; }
	
	m_depth = len();
	if (p)
	   m_depth += p->m_depth;
  }

  ~SuffixNode()
  {
    for (int i = 0; i < basecount; i++)
    {
      if (m_children[i]) { delete m_children[i]; }
    }
  }

  int id()
  {
    if (this) { return m_nodeid; }
    return 0;
  }

  bool isLeaf()
  {
    for (int i = 0; i < basecount; i++)
    {
      if (m_children[i]) { return false; }
    }

    return true;
  }

  const char * str(const char * refstr)
  {
    return substr(refstr, m_start, m_end-m_start+1);
  }

  int len(int i=-1)
  {
    if (i != -1)
    {
      if (i < m_end)
      {
        return i - m_start + 1;
      }
    }

    return m_end - m_start + 1;
  }

	  int depth()
	  { 
		 return m_depth;
	  }

  ostream & printLabel(ostream & os, const char * refstr)
  {
    if (m_start == m_end && m_start == 0)
    {
      os << "\"ROOT\"";
    }
    else
    {
      os << "\"" << str(refstr) << "\"";

       //  << " [" << m_start 
       //  << ","  << m_end 
       //  << "(" << m_nodeid << ")\"";
    }

    return os;
  }


  ostream & printNodeLabel(ostream & os)
  {
    os << m_nodeid;
    return os;
  }

  ostream & printEdgeLabel(ostream & os, const char * refstr)
  {
    string seq = substr(refstr, m_start, m_end-m_start+1);
    os << "\"" << seq << "\"";
    //os << "\"" << seq << " [" << m_start << "," << m_end << "]\"";
    return os;
  }

  int  m_start;                         // start pos in string
  int  m_end;                           // end pos in string
  int  m_nodeid;                        // the id for this node
  int  m_leafid;                        // For leafs, the start position of the suffix in the string
  SuffixNode * m_children [basecount];  // children nodes
  SuffixNode * m_parent;                // parent node
  SuffixNode * m_suffix;                // suffixlink
	  int m_depth;
#if VERIFY
  string m_pathstring;                  // string of path to node
#endif
};

int SuffixNode::s_nodecount(0);

ostream & operator<< (ostream & os, SuffixNode * n)
{
  return n->printNodeLabel(os);
}


// Encapsulate the tree with some helper functions
class SuffixTree
{
public:
  SuffixTree(const char * s) : m_string(s)
  { 
    m_strlen = strlen(s);
#ifdef MPOOL
    m_root = new (&m_pool) SuffixNode(0,0,0,NULL,NULL); // whole tree
#else
    m_root = new SuffixNode(0,0,0,NULL,NULL); // whole tree
#endif
    m_root->m_suffix = m_root;
  }

  ~SuffixTree()
  {
#ifdef MPOOL
#else
	 delete m_root;
#endif
  }

  SuffixNode * m_root;
  const char * m_string;
  int m_strlen;

#ifdef MPOOL
  PoolMalloc_t m_pool;
#endif

  // Print a node for dot
  void printNodeDot(SuffixNode * node, ostream & dfile)
  {
    int children = 0;
    for (int i = 0; i < basecount; i++)
    {
      SuffixNode * child = node->m_children[i];
      if (child)
      {
        children++;

        dfile << " " << node << "->" << child;

        //node->printNodeLabel(dfile, m_string) << " -> ";
        //child->printNodeLabel(dfile, m_string);

        //dfile << " [minlen=" << child->len() << ", label=";
        dfile << " [minlen=1, label=";
        child->printEdgeLabel(dfile, m_string) << "]" << endl;

        printNodeDot(child, dfile);
      }
    }

    if (node->m_suffix)
    {
      dfile << " " << node << " -> " << node->m_suffix
           << " [style=dotted, constraint=false]" << endl;

      //node->printLabel(dfile, m_string) << " -> ";
      //node->m_suffix->printLabel(dfile, m_string) << " [style=dotted, constraint=false]" << endl;
    }

    if (children == 0)
    {
      //dfile << " " << node << " [shape=box, label=";
      //node->printLabel(dfile, m_string) << "]" << endl;

      dfile << " " << node << " [shape=box,width=.2,height=.2,label=\"" << node->id() << ":" << node->m_leafid << "\"]" << endl;
    }
    else
    {
      //dfile << " " << node << " [label=";
      //node->printLabel(dfile, m_string) << "]" << endl;
      dfile << " " << node << " [width=.2,height=.2,label=\"" << node->id() << "\"]" << endl;
    }
  }

  // Print the whole tree for dot
  void printDot(const char * dotfilename)
  {
    ofstream dfile;
    dfile.open(dotfilename, ofstream::out | ofstream::trunc);

    cerr << "Printing dot tree to " << dotfilename << endl;

    dfile << "digraph G {" << endl;
    dfile << " size=\"7.5,10\"" << endl;
    dfile << " center=true" << endl;
    dfile << " label=\"Suffix tree of \'" << m_string << "\' len:" 
          << m_strlen-1 << " nc:"
          << SuffixNode::s_nodecount << "\"" << endl;

    printNodeDot(m_root, dfile);
    dfile << "}" << endl;
  }

  // Print a node in text format
  void printNodeText(ostream & out, SuffixNode * n, int depth)
  {
    for (int b = 0; b < basecount; b++)
    {
      if (n->m_children[b])
      {
        for (int i = 0; i < depth; i++)
        {
          out << " ";
        }
        out << " ";
        out << n->m_children[b]->str(m_string) << endl;
        printNodeText(out, n->m_children[b], depth+1);
      }
    }
  }

  // Print the tree in Text
  void printText(ostream & out)
  {
    out << "Suffix Tree len=" << m_strlen-1 << endl; 
    out << "String: \"" << m_string << "\"" << endl;
    out << "+" << endl;
    printNodeText(out, m_root, 0);
  }

  // Print the tree as list of sorted suffixes
  void printTreeSorted(ostream & out, SuffixNode * node, const string & pathstring)
  {
    bool isLeaf = true;

    string ps(pathstring);
    if (node != m_root) { ps.append(node->str(m_string)); }

    for (int i = 0; i < basecount; i++)
    {
      if (node->m_children[i])
      {
        isLeaf = false;
        printTreeSorted(out, node->m_children[i], ps);
      }
    }

    if (isLeaf) { out << ps << endl; }
  }

  void printTreeFlat(ostream & out)
  {
    cerr << "nodeid\tparent\tSL\tstart\tend\t$\tA\tC\tG\tT\tnodestring" << endl;
    cout << "0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0" << endl;
    printNodeFlat(out, m_root);
  }

  void printNodeFlat(ostream & out, SuffixNode * node)
  {
    out << node->id()           << "\t"
        << node->m_parent->id() << "\t"
        << node->m_suffix->id() << "\t"
        << node->m_start        << "\t"
        << node->m_end          << "\t";

    for (int i = 0; i < basecount; i++)
    {
      out << node->m_children[i]->id() << "\t";
    }

	out << node->m_start << "\t" << node->m_end << endl; 

    if (node == m_root) { out << "ROOT" << endl; } 
    else                { out << node->str(m_string) << endl; }

    for (int i = 0; i < basecount; i++)
    {
      if (node->m_children[i]) { printNodeFlat(out, node->m_children[i]); }
    }
  }

#if VERIFY
  void setNodePath(SuffixNode * node, const string & parentString)
  {
    node->m_pathstring = parentString;

    if (node != m_root)
    {
      node->m_pathstring.append(m_string, node->m_start, node->m_end - node->m_start + 1);
    }

    for (int b = 0; b < basecount; b++)
    {
      if (node->m_children[b])
      {
        setNodePath(node->m_children[b], node->m_pathstring);
      }
    }
  }

  int verifyNodeSuffixLinks(SuffixNode * node, int & linkcount)
  {
    int errs = 0;
    if (node != m_root && node->m_suffix)
    {
      const string & np = node->m_pathstring;
      const string & sp = node->m_suffix->m_pathstring;
      if (np.substr(1, np.length() -1) != sp)
      {
        cerr << "Suffix Link Mismatch!!" << endl;
        node->printLabel(cerr, m_string) << ": " << np << endl;
        node->m_suffix->printLabel(cerr, m_string) << ": " << sp << endl;
        errs++;
      }

      linkcount++;
    }

    if (node == m_root && node->m_suffix != m_root)
    {
      cerr << "Error m_root suffix != m_root !!!" << endl;
      errs++;
    }

    int childcount = 0;
    for (int b = 0; b < basecount; b++)
    {
      if (node->m_children[b])
      {
        childcount++;
        errs += verifyNodeSuffixLinks(node->m_children[b], linkcount);
      }
    }

    if (childcount && !node->m_suffix)
    {
      errs++;
      node->printLabel(cerr, m_string) << " has no suffix link!!!" << endl;
    }

    return errs;
  }

  void verifySuffixLinks()
  {
    cerr << endl;
    cerr << "Verifing links" << endl;
    setNodePath(m_root, "");
    int linkcount = 0;
    int err = verifyNodeSuffixLinks(m_root, linkcount);
    cerr << err << " suffix link errors detected" << endl;
    cerr << linkcount << " suffix links checked" << endl;

    if (err) { exit(1); }
  }
#endif
  

  void buildUkkonen()
  {
    int len = m_strlen - 1; // length of the string, not of the buffer (remove s)
    char base = m_string[1];

    if (DEBUG)
    {
      cerr << "Building Ukkonen Tree for " << m_string << endl
           << "Len: " << len << endl;
    }

    // Construct T1
#ifdef MPOOL
    SuffixNode * node = new (&m_pool) SuffixNode(1, len, 1, m_root, NULL); // leaf: 1
#else
    SuffixNode * node = new SuffixNode(1, len, 1, m_root, NULL); // leaf: 1
#endif
    m_root->m_children[b2i(base)] = node;
    SuffixNode * firstleaf = node;
    SuffixNode * lastleaf = node;

    if (DEBUG)
    { cerr << "Phase 1 Child: "; node->printLabel(cerr, m_string) << endl; }

    int startj = 2;

    // phase i+1
    for (int i = 2; i <= len; i++)
    {
      // Start at the last leaf created which will allow easy
      // access to the node for startj
      node = lastleaf;
      int nodewalk = 0;

      // Keep track of last internal nodes created in split so we can add suffix links
      SuffixNode * splitnode = NULL;

      if (!DOPHASETRICK)
      {
        startj = 2;
        node = firstleaf;
      }

      if (DEBUG) 
      { 
        char next = m_string[i];
        cerr << endl;
        cerr << i << ".0 " << "Phase " << i << " adding " << next << " starting with " << startj << endl; 

        string beta = substr(m_string, 1, i);
        cerr << i << ".1" << " Extension 1: \"" << beta << "\" [implicit]" << endl;
      }

      for (int j = startj; j <= i; j++)
      {
        // Goal: Ensure S[j .. i] (beta) is in the suffix tree 
        // Precondition: S[j-1 .. i] (alpha) is in the suffix tree "near" node
        //               All Internal nodes have a suffix link

        // Idea: 1) Remember where alpha is in the tree relative to node
        //       2) Walk up the tree w bases until we get to a node with a suffix link.
        //       3) Follow suffix link which shifts the path from S[j-1..i] to S[j..i]
        //       4) Walk down tree in new location ensuring S[i-w .. i] is in tree

        // Notes: 1) All internal nodes have a suffix link by next extension
        //        2) Any time we walk up to root, have to check S[j..i]
        //        3) Suffix [1..i] is always present so start extension j with 2

        int betapos = i; // The first position in string we need to check in tree

        if (DEBUG)
        {
          cerr << endl;
          string beta = substr(m_string, j, i-j+1);
          cerr << i << "." << j << " Phase " << i << " Extension " << j << ": \"" << beta << "\" bp:" << betapos << endl;

          cerr << i << "." << j << "  Walking up from n:"; 
          node->printLabel(cerr, m_string) << " nw: " << nodewalk << endl;
        }

        if (node == m_root)
        {
          // If we are at root, we have to check the full string s[j..i] anyways
        }
        else
        {
          if (nodewalk)
          {
            // partially walked down node->child, but didn't switch to child
            // Match at i=6 on left... nodewalk=2, at 5 after suffix link
            // 5 = i-2+1
            //                 o ----- o
            //               5 A       A 5  <-
            //            -> 6 T       T 6 

            betapos -= nodewalk-1;

            if (DEBUG)
            {
              cerr << i << "." << j << "   Adjusted nw: " << nodewalk << endl;
            }
          }
          else
          {
            // Exactly at a node or leaf. 
            // Walk up to parent, subtracting length of that edge
            int len = node->len(i);
            betapos -= len-1;
            node = node->m_parent;

            if (DEBUG)
            {
              cerr << i << "." << j << "   Adjusted len: " << len << endl;
            }
          }
          
          if (DEBUG)
          {
            cerr << i << "." << j << "   parent bp: " << betapos <<  " n:";
            node->printLabel(cerr, m_string) << endl;
          }

          if (node->m_suffix == NULL)
          {
            // Subtract entire edge length
            betapos -= node->len(i);
            node = node->m_parent;

            if (DEBUG)
            {
              cerr << i << "." << j << "   grandparent bp: " << betapos << " n:";
              node->printLabel(cerr, m_string) << endl;
            }

            #if VERIFY
            if (node->m_suffix == NULL)
            {
              cerr << "Missing suffix link!!! ";
              exit(1);
            }
            #endif
          }
        }

        // jump across suffix link
        node = node->m_suffix;
        if (node == m_root) { betapos = j; } // have to check full string

        if (DEBUG)
        {
          cerr << i << "." << j << "  Starting to walk down from bp: " << betapos << " to " << i << " n:";
          node->printLabel(cerr, m_string) << endl;
        }

        if (FORCEROOT && node != m_root)
        {
          node = m_root;
          betapos = j;

          if (DEBUG)
          {
            cerr << i << "." << j << " AtRoot bp: " << betapos << endl;
          }
        }

        bool done = false;
        startj = j+1; // assume this extension should be skipped in the next phase

        while ((betapos <= i) && !done)
        {
          char base = m_string[betapos];
          unsigned char b = b2i(base);
          SuffixNode * child = node->m_children[b];

          if (DEBUG)
          {
            cerr << i << "." << j << "  node betapos: " << betapos << "[" << base << "] n:";
            node->printLabel(cerr, m_string) << " ";
            if (child) { cerr << "c: "; child->printLabel(cerr, m_string); } 
            cerr << endl;
          }

          if (!child)
          {
            if (splitnode && betapos == splitnode->m_start)
            {
              if (DEBUG)
              {
                cerr << i << "." << j << "   Add SL1: ";
                splitnode->m_parent->printLabel(cerr, m_string) << " sl-> ";
                node->printLabel(cerr, m_string) << endl;
              }

              splitnode->m_parent->m_suffix = node;
              splitnode = NULL;
            }

#ifdef MPOOL
            SuffixNode * newnode = new (&m_pool) SuffixNode(betapos, len, j, node, NULL); // leaf: j
#else
            SuffixNode * newnode = new SuffixNode(betapos, len, j, node, NULL); // leaf: j
#endif
            node->m_children[b] = newnode; 
            lastleaf = newnode;

            if (DEBUG)
            {
              cerr << i << "." << j << "   New Node: ";
              newnode->printLabel(cerr, m_string) << endl;
            }

            node = newnode;

            // This is the first base that differs, but the edgelength to 
            // i may be longer. Therefore set nodewalk to 0, so the entire
            // edge is subtracted.
            nodewalk = 0;
            done = true;
            break;
          }
          else
          {
            int nodepos = child->m_start;
            nodewalk = 0;

            char nodebase = m_string[nodepos];

            #if VERIFY
            if (nodebase != base)
            {
              char nb = m_string[nodepos];
              cerr << "ERROR: first base on edge doesn't match edge label" << endl;
              cerr << "       nb: " << nb << " base: " << base << endl;
              exit(1);
            }
            #endif

            // By construction, the string from j-1 to betapos to i-1
            // must already by present in the suffix tree
            // Therefore, we can skip checking every character, and zoom
            // to exactly the right character, possibly skipping the entire edge

            if (DOJUMP)
            {
              int mustmatch = i-1 - betapos + 1;
              int childlen = child->len(i);

              if (mustmatch >= childlen)
              {
                betapos += childlen;
                nodepos += childlen;

                skippedbases += childlen;

                if (DEBUG)
                {
                  cerr << i << "." << j << "   Edge Jump by: " << childlen << " new bp: " << betapos << " np: " << nodepos << endl;
                }

                #if VERIFY
                if (nodepos != child->m_end+1)
                {
                  cerr << "ERROR: jump should have skipped entire edge, but didn't!" << endl;
                  exit(1);
                }
                #endif
              }
              else if (mustmatch)
              {
                betapos += mustmatch;
                nodepos += mustmatch;
                nodewalk += mustmatch;

                skippedbases += mustmatch;

                if (DEBUG)
                {
                  cerr << i << "." << j << "   Partial Jump by: " << mustmatch << " new bp: " << betapos << " np: " << nodepos << endl;
                }

                #if VERIFY
                if (VERIFY)
                {
                  if (m_string[betapos-1] != m_string[nodepos-1])
                  {
                    cerr << "ERROR: jump should have matched at least the mustmatch-1 characters" << endl;
                    cerr << "s[bp-1]: " << m_string[betapos-1] << " s[np-1]: " << m_string[nodepos-1] << endl;
                    exit(1);
                  }
                }
                #endif
              }
            }

            while (nodepos <= child->m_end && betapos <= i)
            {
              nodebase = m_string[nodepos];

              #if VERBOSE
                cerr << i << "." << j << "   child bp: " << betapos << "[" << m_string[betapos] 
                     << "] nb [" << nodebase << "]" << endl;
              #endif

              if (m_string[betapos] == nodebase)
              {
                if (splitnode && betapos == splitnode->m_start)
                {
                  if (DEBUG)
                  {
                    cerr << i << "." << j << "   Add SL2: ";
                    splitnode->m_parent->printLabel(cerr, m_string) << " sl-> ";
                    node->printLabel(cerr, m_string) << endl;
                  }

                  splitnode->m_parent->m_suffix = node;
                  splitnode = NULL;
                }

                nodepos++; betapos++; nodewalk++;

                if (betapos == i+1)
                {
                  if (DEBUG)
                  {
                    cerr << i << "." << j << "    Internal edge match nw: " << nodewalk << endl;
                  }

                  if ((nodewalk == child->len(i)) && (child->m_end == len))
                  {
                    // we walked the whole edge to leaf, implicit rule I extension
                    if (DEBUG)
                    {
                      cerr << i << "." << j << "    Leaf Node, Implicit Rule I Extension" << endl;
                    }
                  }
                  else
                  {
                    // "Real" rule III implicit extension

                    // The j-1 extension was the last explicit extension in this round
                    // Start the next round at the last explicit extension
                    if (DOPHASETRICK)
                    {
                      startj = j;

                      int skip = startj - 2;

                      if (DEBUG)
                      {
                        cerr << i << "." << j << "    Implicit Extension... start next phase at " << startj << ", saved " << skip << endl;
                      }

                      skippedextensions += skip;
                    }

                    if (DOINTERNALSKIP)
                    {
                      // Since we hit an internal match on a non-leaf, we know every other 
                      // extension in this phase will also hit an internal match. 

                      // Have to be careful since leafs get the full string immediately, but
                      // they really have a Rule 1 extension

                      int skip = i-j;

                      if (DEBUG)
                      { 
                        cerr << i << "." << j << "    Implicit Extension... skipping rest of phase, saved " << skip << endl;
                      }

                      skippedextensions += skip;
                      j = i+1;
                    }
                  }

                  done = true;
                }
              }
              else
              {
                if (DEBUG) { cerr << i << "." << j << "   Spliting "; child->printLabel(cerr, m_string); }

                // Split is a copy of the child with the end shifted
                // Then adjust start of child
#ifdef MPOOL
                SuffixNode * split = new (&m_pool) SuffixNode(child->m_start, nodepos-1, 0, node, NULL); // internal
#else
                SuffixNode * split = new SuffixNode(child->m_start, nodepos-1, 0, node, NULL); // internal
#endif

                split->m_children[b2i(nodebase)] = child;
                child->m_start = nodepos;
                child->m_parent = split;

                if (DEBUG)
                {
                  cerr << " => ";
                  split->printLabel(cerr, m_string) << " + ";
                  child->printLabel(cerr, m_string) << endl;
                }

                node->m_children[b] = split;
                node = split;

                if (splitnode && betapos == splitnode->m_start)
                {
                  if (DEBUG)
                  {
                    cerr << i << "." << j << "   Add SL3: ";
                    splitnode->m_parent->printLabel(cerr, m_string) << " sl-> ";
                    node->printLabel(cerr, m_string) << endl;
                  }

                  splitnode->m_parent->m_suffix = split;
                  splitnode = NULL;
                }

                // Now create the new node
#ifdef MPOOL
                SuffixNode * newnode = new (&m_pool) SuffixNode(betapos, len, j, split, NULL); // leaf j
#else
                SuffixNode * newnode = new SuffixNode(betapos, len, j, split, NULL); // leaf j
#endif
                lastleaf = newnode;

                split->m_children[b2i(m_string[betapos])] = newnode; 
                splitnode = newnode;

                node = newnode;

                if (DEBUG)
                {
                  cerr << i << "." << j << "   Split New Node: ";
                  newnode->printLabel(cerr, m_string) << endl;
                }

                // This is the first base that differs, but the edgelength to 
                // i may be longer. Therefore set nodewalk to 0, so the entire
                // edge is subtracted.
                nodewalk = 0;
                done = true;
                break;
              }
            }
          }

          if (!done) { node = child; }
        }
      }

      #if VERIFY
      if (VERIFY) { verifySuffixLinks(); }
      #endif
    }
  }
};


SuffixTree * gtree = NULL;

void buildUkkonenSuffixTree(const char * str)
{
   gtree = new SuffixTree(str);
   gtree->buildUkkonen();
}

static const int MAX_TEXTURE_DIMENSION = 4096;
static const int BLOCKSIZE = 32;

inline TextureAddress id2addr(int id)
{
  TextureAddress retval;

  int bigx = id & 0x1FFFF;
  int bigy = id >> 17; 
  retval.y = (bigy << 5) + (bigx & 0x1F); 
  retval.x = bigx >> 5; 

  return retval;
}

void buildNodeTexture(SuffixNode * node, 
                      PixelOfNode * nodeTexture, 
                      PixelOfChildren * childrenTexture,
                      AuxiliaryNodeData aux_data[],
                      const char * refstr)
{	
  int id = node->id();

  aux_data[id].length = node->len();
  aux_data[id].depth  = node->depth();
  aux_data[id].leafid = node->m_leafid;
  aux_data[id].parent = id2addr(node->m_parent->id());
 
  if (aux_data[id].leafid != 0)
  {
    aux_data[id].leafchar = refstr[aux_data[id].leafid-1];
  }
  else
  {
    aux_data[id].leafchar = 0;
  }

  TextureAddress myaddress(id2addr(id));
  id = myaddress.x + myaddress.y*MAX_TEXTURE_DIMENSION;

  nodeTexture[id].start   = node->m_start;
  nodeTexture[id].end     = node->m_end;
  nodeTexture[id].suffix  = id2addr(node->m_suffix->id());


  for (int i = 0; i < basecount; i++)
  {
    if (node->m_children[i]) 
    { 
      TextureAddress childaddr = id2addr(node->m_children[i]->id());

      // Unfortunately, the $ link doesn't fit into PixelOfChildren
      if (i == b2i('$')) 
      { 
         nodeTexture[id].childD          = childaddr; 
      }
      else               
      { 
         childrenTexture[id].children[i] = childaddr; 
      }
      
      buildNodeTexture(node->m_children[i], nodeTexture, childrenTexture, aux_data, refstr); 
    }
  }
}

void buildSuffixTreeTexture(PixelOfNode** nodeTexture, 
                            PixelOfChildren **childrenTexture, 
                            unsigned int* width, unsigned int* height,
                            AuxiliaryNodeData **aux_data,
                            const char * refstr)
{
	assert(SuffixNode::s_nodecount < MAX_TEXTURE_DIMENSION*MAX_TEXTURE_DIMENSION);
    assert(sizeof(PixelOfNode) == 16);
    assert(sizeof(PixelOfChildren) == 16);

    // Leave space for NULL node
    int allnodes = SuffixNode::s_nodecount+1;

    *width = MAX_TEXTURE_DIMENSION;
    *height = (int)ceil((allnodes+0.0) / MAX_TEXTURE_DIMENSION)+BLOCKSIZE;

    // allocate space for the node and children textures
    *nodeTexture     = (PixelOfNode*)     calloc((*width)*(*height), sizeof(PixelOfNode));
    *childrenTexture = (PixelOfChildren*) calloc((*width)*(*height), sizeof(PixelOfChildren));

    *aux_data = (AuxiliaryNodeData*)calloc(SuffixNode::s_nodecount + 1, sizeof(AuxiliaryNodeData));

    if (!*nodeTexture || !*childrenTexture || !*aux_data)
    {
      printf("arg.  texture allocation failed.\n");
      exit(-1);
    }

    buildNodeTexture(gtree->m_root, *nodeTexture, *childrenTexture, *aux_data, refstr);
};


void printTreeTexture(const char * texfilename,
                      PixelOfNode * nodeTexture,
                      PixelOfChildren * childrenTexture,
                      int nodecount)
{
  cerr << "Printing tree texture to " << texfilename << endl;

  ofstream texfile;
  texfile.open(texfilename, ofstream::out | ofstream::trunc);

  texfile << "id\tx\ty\tstart\tend\ta.x\ta.y\tc.x\tc.y\tg.x\tg.y\tt.x\tt.y\t$.x\t$.y" << endl;
  for (int i = 0; i < nodecount; i++)
  {
    TextureAddress myaddress(id2addr(i)); 

    texfile << i << "\t"
            << myaddress.x << "\t"
            << myaddress.y << "\t"
            << nodeTexture[i].start << "\t"
            << nodeTexture[i].end   << "\t";

    for (int j = 0; j < 4; j++)
    {
      texfile << childrenTexture[i].children[j].x << "\t";
      texfile << childrenTexture[i].children[j].y << "\t";
    }

    texfile << nodeTexture[i].childD.x << "\t";
    texfile << nodeTexture[i].childD.y << endl;
  }

  texfile.close();
}

void renumberTree()
{
  queue<pair<SuffixNode *, int> > nodequeue;

  nodequeue.push(make_pair(gtree->m_root,0));
  int nodecount = 0;

  while(!nodequeue.empty())
  {
    pair<SuffixNode *,int> npair = nodequeue.front(); nodequeue.pop();

    SuffixNode * node = npair.first;
    int depth = npair.second;
    
    node->m_nodeid = ++nodecount;

    if (depth < 16)
    {
      for (int i = 0; i < basecount; i++)
      {
        SuffixNode * child = node->m_children[i];
        if (child) { nodequeue.push(make_pair(child,depth+1)); }
      }
    }
    else
    {
      for (int i = 0; i < basecount; i++)
      {
        SuffixNode * child = node->m_children[i];
        if (child)
        {
          child->m_nodeid = ++nodecount;
          
          for(int j = 0; j < basecount; j++)
          {
            SuffixNode * gchild = child->m_children[j];

            if (gchild)
            {
              gchild->m_nodeid = ++nodecount;
              for (int k = 0; k < basecount; k++)
              {
                SuffixNode * ggchild = gchild->m_children[k];

                if (ggchild)
                { 
                  ggchild->m_nodeid = ++nodecount;

                  for (int l = 0; l < basecount; l++)
                  {
                    SuffixNode * gggchild = ggchild->m_children[l];

                    if (gggchild)
                    {
                      gggchild->m_nodeid = ++nodecount;

                      for (int m = 0; m < basecount; m++)
                      {
                        SuffixNode * ggggchild = gggchild->m_children[m];
                        if (ggggchild){ nodequeue.push(make_pair(ggggchild, depth+5)); }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


extern "C"
void createTreeTexture(const char * refstr, 
                       PixelOfNode** nodeTexture, 
                       PixelOfChildren** childrenTexture, 
                       unsigned int* width, unsigned int* height,
					   AuxiliaryNodeData** aux_data,
					   int* num_nodes,
                       const char * dotfilename,
                       const char * texfilename)
{
  cerr << "  Creating Suffix Tree... ";
  EventTime_t btimer;
  SuffixNode::s_nodecount = 0;
  buildUkkonenSuffixTree(refstr); 
  cerr << SuffixNode::s_nodecount << " nodes "
       << btimer.str(true, 5) << endl;

  cerr << "  Renumbering tree... ";
  EventTime_t rtimer;
  renumberTree();
  cerr << rtimer.str(true, 5) << endl;


  EventTime_t ftimer;
  cerr << "  Flattening Tree... ";
  // indexTree();
  buildSuffixTreeTexture(nodeTexture, 
                         childrenTexture, 
                         width, height,
                         aux_data,
                         gtree->m_string);

  *num_nodes = SuffixNode::s_nodecount;
  cerr << ftimer.str(true, 5) << endl;

  if (dotfilename)
  {
    gtree->printDot(dotfilename);
  }

  if (texfilename)
  {
    printTreeTexture(texfilename, *nodeTexture, *childrenTexture, SuffixNode::s_nodecount+1);
  }

  delete gtree;
  gtree = NULL;
}


extern "C"
void getReferenceString(const char * filename, char** refstr, size_t* reflen)
{
  EventTime_t timer;
  cerr << "Loading ref: " << filename << "... ";

  string S="s";

  ifstream file;
  file.open(filename);

  if (!file)
  {
    cerr << "Can't open " << filename << endl;
    exit (1);
  }

  // Skip over the reference header line
  char refline[2048];
  file.getline(refline, sizeof(refline));

  if (refline[0] != '>')
  {
    cerr << endl
         << "ERROR: Reference file is not in FASTA format"
         << endl;
  }

  // Now read the reference string
  string buffer;
  while (file >> buffer)
  {
    if (buffer[0] == '>')
    {
      cerr << endl
           << "ERROR: Only a single reference sequence is supported!" 
           << endl;

      exit (1);
    }
    else
    {
      for (unsigned int i = 0; i < buffer.length(); i++)
      {
        char b = toupper(buffer[i]);
        if (b == 'A' || b == 'C' || b == 'G' || b=='T')
        {
          S += b;
        }
		else
		{
		   S += 'A';
		}
      }
    }
  }

  S += "$";

  *refstr = strdup(S.c_str());
  *reflen = strlen(*refstr) + 1;

  cerr << *reflen-3 << " bp. " << timer.str(true, 5) << endl;
}

inline void addChar(char **buf, int * size, int * pos, char c)
{
  if (*pos == *size)
  {
    (*size) *= 2; // double the size of the buffer
    *buf = (char *) realloc(*buf, *size);
    if (!*buf)
    {
      cerr << "ERROR: Realloc failed, requested: " << *size << endl;
    }
  }

  (*buf)[*pos] = c;
  (*pos)++;
}

inline size_t bytesNeededOnGPU(unsigned int querylen, int min_match_len)
{
   if (min_match_len == -1)
	  return sizeof(MatchCoord) + (querylen + 10);
   else
	  return sizeof(MatchCoord) * (querylen - min_match_len + 1) + (querylen + 10);
} 

//Gets up to set_size queries. 
extern "C"
void getQueriesTexture(int qfile,
                       char** queryTexture, 
                       size_t* queryTextureSize, 
                       int** queryAddrs, 
					   char*** queryNames,
					   int** queryLengths,
					   unsigned int* numQueries,
					   size_t memory_avail,
					   int min_match_length,
					   bool rc)
{
  EventTime_t timer;
//fprintf(stderr,"1");
  int qstringpos = 0;
  int qstringsize = 1024*1024;
  char * qstring = (char *) malloc(qstringsize);

  bool resetAmbiguity  = true;

  // offset of query i in qstring
  int offsetspos = 0;
  int offsetssize = 1024;
  int * offsets = (int *) malloc(offsetssize * sizeof(int));
  int * lengths = (int *) malloc(offsetssize * sizeof(int));

  int qrylen = 0;
  int this_qrylen = 0;

  int bytes_read;
  unsigned char buf[32*1024];
  
  vector<char*> names;
  string header;
  bool inheader = false;
  int total_read = 0;

  unsigned char dnachar [256];

  bool set_full = false;
//fprintf(stderr,"2");
  // tracks the GPU memory needed by the queries read so far.
  size_t curr_mem_usage = 0;

  for (int i = 0; i < 256; i++)
  {
    dnachar[i] = 0;
  }

  dnachar[(unsigned char) 'A'] = 1;
  dnachar[(unsigned char) 'C'] = 1;
  dnachar[(unsigned char) 'G'] = 1;
  dnachar[(unsigned char) 'T'] = 1;
//fprintf(stderr,"3");
  while ((bytes_read = read(qfile, buf, sizeof(buf))) != 0)
  {
   // cerr << "bytes_read: " << bytes_read << endl;

    if (bytes_read == -1)
    {
      cerr << "ERROR: Error reading file: " << errno << endl;
      exit(1);
    }

    int i = 0;

    if (inheader)
    {
      // Handle case where last read was inside a header
	   for (; i < bytes_read; i++)
	   {
          if (buf[i] == '\n')
          {
			 inheader = false;
			 i++;
			 char* name = strdup(header.c_str());
			 names.push_back(name);
			 header.clear();
			 break;
          }
		  else
		  {
			 header.insert(header.end(), buf[i]);			 
		  }
        }
    }
//  fprintf(stderr,"4");
    for (; i < bytes_read; i++)
    {
      unsigned char b = toupper(buf[i]);

      if (b == '>')
      {

		if (curr_mem_usage >= memory_avail)
		{
		   set_full = true;
		   off_t seek = lseek(qfile, -(bytes_read - i), SEEK_CUR);
		   if (seek == (off_t)-1)
		   {
			  cerr<< "lseek failed: "<< errno<<endl;
			  exit(-1);
		   }
		   break;
		}
//  fprintf(stderr,"5");

        // in a header line
        if (offsetspos != 0) 
        {  
		   if (this_qrylen < min_match_length)
		   {
			  printf("> %s\n", names.back());
			  if (rc)
				 printf("> %s Reverse\n", names.back());
			  names.pop_back();
			  --offsetspos;	
			  qstringpos -= this_qrylen  + 1;
		   }
		   else
		   {
			  addChar(&qstring, &qstringsize, &qstringpos, '\0');
			  lengths[offsetspos - 1] = this_qrylen;
			  curr_mem_usage += bytesNeededOnGPU(this_qrylen, min_match_length);
		   }
        }
//  fprintf(stderr,"6");
        if (offsetspos == offsetssize)
        {
          offsetssize *= 2;
          offsets = (int *) realloc(offsets, sizeof(int)*offsetssize);
		  lengths = (int *) realloc(lengths, sizeof(int)*offsetssize);
          if (!offsets || !lengths)
          {
            cerr << endl
                 << "ERROR: Realloc failed: requested " 
                 << sizeof(int) * offsetssize << endl;
            exit(1);
          }
        }

        offsets[offsetspos++] = qstringpos;
        
        inheader = true;

        // Try to walk out of header
        for (i++; i < bytes_read; i++)
        {
          if (buf[i] == '\n')
          {
			  inheader = false;
			  char* name = strdup(header.c_str());
			  names.push_back(name);
			  header.clear();
			  break;
          }
		  else
		  {
			 header.insert(header.end(), buf[i]);
		  }
        }

		addChar(&qstring, &qstringsize, &qstringpos, 'q');
		this_qrylen = 0;
      }
      else if (dnachar[b])
      {
        addChar(&qstring, &qstringsize, &qstringpos, b);
        qrylen++;
		this_qrylen++;
      }
      else if (isspace(b))
      {

      }
      else if (resetAmbiguity)
      {
        addChar(&qstring, &qstringsize, &qstringpos, 'x');
		this_qrylen++;
      }
      else
      {
        cerr << endl
             << "ERROR: Unexpected character: " << buf[i] 
             << " in query file at byte: " << total_read+i << endl;
        exit(1);
      }
    }
// fprintf(stderr,"7");
	if (set_full)
	   break;

    total_read += bytes_read;
  }

  if (qstringpos) 
  { 
	 if (this_qrylen < min_match_length)
	 {
		printf("> %s\n", names.back());
		if (rc)
		   printf("> %s Reverse\n", names.back());
		names.pop_back();
		--offsetspos;
		qstringpos -= this_qrylen + 1;
	 }
	 else
	 {
		addChar(&qstring, &qstringsize, &qstringpos, '\0');
		lengths[offsetspos - 1] = this_qrylen;
		curr_mem_usage += bytesNeededOnGPU(this_qrylen, min_match_length);
	 }
  }

  *numQueries = offsetspos;

  if (offsetspos == 0)
  {
	 free(offsets);
	 free(lengths);
	 free(qstring);
	 *queryAddrs = NULL;
	 *queryTexture = NULL;
	 *queryTextureSize = 0;
	 *queryNames = NULL;
	 
	 return;
  }

  
  *queryAddrs = offsets;

  *queryTexture = qstring;
  *queryTextureSize = qstringpos;
  *queryNames = (char**)malloc(names.size() * sizeof(char*));
  *queryLengths = lengths;
//fprintf(stderr,"8");
  for (unsigned int i = 0; i < *numQueries; ++i)
  {
	 *(*queryNames + i) = names[i];
  }
  cerr << offsetspos << " queries (" 
       << qrylen << " bp), need " 
	   << curr_mem_usage << " bytes on the GPU " 
       << timer.str(true, 5) << endl;
}

struct pathblock
{
	  TextureAddress node_addr;
	  int string_depth;
};

#define __USE_BUFFERED_IO__ 

static const size_t output_buf_limit = 32*1024;
char output_buf[output_buf_limit];

//FIXME: needs to be reinitialized to zero at the beginning of each round of printing.
size_t bytes_written = 0;

int addToBuffer(char* string)
{
	 size_t buf_length = strlen(string);
	 
	 if (buf_length + bytes_written>= output_buf_limit)
	 {
		size_t chunk = (output_buf_limit - bytes_written - 1);
		strncpy(output_buf + bytes_written, string, chunk);
		output_buf[bytes_written + chunk] = 0;
		printf("%s", output_buf);
		strncpy(output_buf, string + chunk, buf_length - chunk);
		bytes_written = buf_length - chunk;
	 }
	 else
	 {
		strncpy(output_buf + bytes_written, string, buf_length);
		bytes_written += buf_length;
	 }
   return 0;
}


inline int addr2id(TextureAddress addr)
{
   int blocky = addr.y & 0x1F;
   int bigy = addr.y >> 5;
   int bigx = (addr.x << 5) + blocky;
   return bigx + (bigy << 17);
}

#define CHILDREN(node_addr) ((((PixelOfChildren*)(page->ref.h_children_tex_array)) + (node_addr.x) + ((node_addr.y) * MAX_TEXTURE_DIMENSION))->children)
#define DOLLAR(node_addr)   ((((PixelOfNode*)    (page->ref.h_node_tex_array))     + (node_addr.x) + ((node_addr.y) * MAX_TEXTURE_DIMENSION))->childD)
#define LEAFID(x)                                (page->ref.aux_data[x].leafid)
#define LEAFCHAR(x)                              (page->ref.aux_data[x].leafchar)

char buf[256];

void printNodeAlignments(const char* ref,
						 const ReferencePage* page,
						 const char queryflankingbase, 
						 const TextureAddress node,
						 const int qrypos, 
						 int qrylen,
						 const pathblock path[],
						 int path_idx,
                         bool rc)
{
  int nodeid = addr2id(node);
  char isLeaf = LEAFCHAR(nodeid); 

  if (path[path_idx].node_addr.data == node.data)
  {
	 qrylen = path[path_idx].string_depth;
	 path_idx--;  
  }
  
  if (isLeaf)
  {
     if (isLeaf != queryflankingbase)
     {
       int leafid = LEAFID(nodeid);
       int left_in_ref = (leafid - 1) + page->begin; 
       int right_in_ref = left_in_ref + qrylen;

       if ((left_in_ref != page->begin || page->shadow_left == -1) && 
           (right_in_ref != page->end || page->shadow_right == -1))
       {
		 if (!(left_in_ref > page->begin && right_in_ref < page->shadow_left))
		 {
		   //sprintf(buf, "\t%d\t%d\t%d\n", node->m_leafid, qrypos, qrylen);
		   sprintf(buf, "%8d%10d%10d\n", left_in_ref, qrypos, qrylen);
		   addToBuffer(buf);
		 }
	   }
	 }
  }
  else
  {
    TextureAddress* children = CHILDREN(node);
    for (int i = 0; i < basecount - 1; ++i)
    {
       if ((children + i)->data)
       {
          printNodeAlignments(ref, page, queryflankingbase, *(children+i), 
                              qrypos, qrylen, path, path_idx, rc);
       }
    }
    
    TextureAddress dollar = DOLLAR(node);
    if (dollar.data)
    {
       printNodeAlignments(ref, page, queryflankingbase, dollar, 
                           qrypos, qrylen, path, path_idx, rc);
    }
  }
}

void flushOutput()
{
   if (bytes_written)
   {
	  output_buf[bytes_written] = 0;
	  printf("%s", output_buf);
	  bytes_written  = 0;
   }
}


//FIXME: hardcoded path buffer, needs to be as long as the longest query in the query set.
pathblock path[8192];

#define NODE_SDEPTH(x)  (page->ref.aux_data[x].depth)
#define NODE_LENGTH(x) (page->ref.aux_data[x].length)
#define NODE_PARENT(x) (page->ref.aux_data[x].parent)

char RC(char c)
{
  switch(c)
  {
    case 'A': return 'T';
    case 'C': return 'G';
    case 'G': return 'C';
    case 'T': return 'A';
    case 'q': return '\0';
    default:  return c;
  };
}

void printAlignments(char* ref,
					 ReferencePage* page,
					 char* query, 
                     int qrylen,
					 int nodeid, 
					 int qrypos, 
					 int edge_match, 
					 int min_match,
                     bool rc,
                     bool forwardcoordinates)
{
   TextureAddress node_addr = id2addr(nodeid);
   TextureAddress prev;
   prev.data = 0; 
  
   int path_idx = 0;
   int string_depth = NODE_SDEPTH(nodeid) - 1;

   if (edge_match > 0)
   {
     string_depth = NODE_SDEPTH(nodeid) - (NODE_LENGTH(nodeid) - edge_match) - 1;
   }
   else
   {
     edge_match = NODE_LENGTH(nodeid);
   }

   path[path_idx].node_addr = node_addr;
   path[path_idx].string_depth = string_depth;
   path_idx++;
   string_depth -= edge_match;
   prev = node_addr;

   node_addr = NODE_PARENT(nodeid);
   
   while ((node_addr.data) && string_depth >= min_match)
   {
	  nodeid = addr2id(node_addr);
	  path[path_idx].node_addr = node_addr;
	  path[path_idx].string_depth = string_depth;
	  path_idx++;	
	  string_depth -= NODE_LENGTH(nodeid);	  
	  
	  prev = node_addr;
	  node_addr = NODE_PARENT(nodeid);
   }
   
   char flankingbase = query[qrypos];

   if (rc)
   {
     flankingbase = RC(query[strlen(query)-qrypos]);
     if (forwardcoordinates) { qrypos = qrylen - 1 - qrypos; }
   }

   printNodeAlignments(ref, page, flankingbase, prev, qrypos + 1, 
   				       NODE_SDEPTH(addr2id(prev)), path, path_idx - 1, rc);
}
