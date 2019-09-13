#ifndef __cuobjdump_h__
#define __cuobjdump_h__
#include <iostream>
#include <list>
#include <string>

typedef void *yyscan_t;
struct cuobjdump_parser {
  yyscan_t scanner;
  int elfserial;
  int ptxserial;
  FILE *ptxfile;
  FILE *elffile;
  FILE *sassfile;
  char filename[1024];
};

class cuobjdumpSection {
 public:
  // Constructor
  cuobjdumpSection() {
    arch = 0;
    identifier = "";
  }
  virtual ~cuobjdumpSection() {}
  unsigned getArch() { return arch; }
  void setArch(unsigned a) { arch = a; }
  std::string getIdentifier() { return identifier; }
  void setIdentifier(std::string i) { identifier = i; }
  virtual void print() {
    std::cout << "cuobjdump Section: unknown type" << std::endl;
  }

 private:
  unsigned arch;
  std::string identifier;
};

class cuobjdumpELFSection : public cuobjdumpSection {
 public:
  cuobjdumpELFSection() {}
  virtual ~cuobjdumpELFSection() {
    elffilename = "";
    sassfilename = "";
  }
  std::string getELFfilename() { return elffilename; }
  void setELFfilename(std::string f) { elffilename = f; }
  std::string getSASSfilename() { return sassfilename; }
  void setSASSfilename(std::string f) { sassfilename = f; }
  virtual void print() {
    std::cout << "ELF Section:" << std::endl;
    std::cout << "arch: sm_" << getArch() << std::endl;
    std::cout << "identifier: " << getIdentifier() << std::endl;
    std::cout << "elf filename: " << getELFfilename() << std::endl;
    std::cout << "sass filename: " << getSASSfilename() << std::endl;
    std::cout << std::endl;
  }

 private:
  std::string elffilename;
  std::string sassfilename;
};

class cuobjdumpPTXSection : public cuobjdumpSection {
 public:
  cuobjdumpPTXSection() { ptxfilename = ""; }
  std::string getPTXfilename() { return ptxfilename; }
  void setPTXfilename(std::string f) { ptxfilename = f; }
  virtual void print() {
    std::cout << "PTX Section:" << std::endl;
    std::cout << "arch: sm_" << getArch() << std::endl;
    std::cout << "identifier: " << getIdentifier() << std::endl;
    std::cout << "ptx filename: " << getPTXfilename() << std::endl;
    std::cout << std::endl;
  }

 private:
  std::string ptxfilename;
};

#endif /* __cuobjdump_h__ */
