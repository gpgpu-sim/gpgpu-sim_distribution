// $Id: config_utils.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _CONFIG_UTILS_HPP_
#define _CONFIG_UTILS_HPP_

#include "booksim.hpp"

#include<cstdio>
#include<string>
#include<map>
#include<vector>

extern "C" int yyparse();

class Configuration {
  static Configuration * theConfig;
  FILE * _config_file;
  string _config_string;

protected:
  map<string,string> _str_map;
  map<string,int>    _int_map;
  map<string,double> _float_map;
  
public:
  Configuration();

  void AddStrField(string const & field, string const & value);

  void Assign(string const & field, string const & value);
  void Assign(string const & field, int value);
  void Assign(string const & field, double value);

  string GetStr(string const & field) const;
  int GetInt(string const & field) const;
  double GetFloat(string const & field) const;

  vector<string> GetStrArray(const string & field) const;
  vector<int> GetIntArray(const string & field) const;
  vector<double> GetFloatArray(const string & field) const;

  void ParseFile(string const & filename);
  void ParseString(string const & str);
  int  Input(char * line, int max_size);
  void ParseError(string const & msg, unsigned int lineno = 0) const;
  
  void WriteFile(string const & filename);
  void WriteMatlabFile(ostream * o) const;

  inline const map<string, string> & GetStrMap() const {
    return _str_map;
  }
  inline const map<string, int> & GetIntMap() const {
    return _int_map;
  }
  inline const map<string, double> & GetFloatMap() const {
    return _float_map;
  }

  static Configuration * GetTheConfig();

};

bool ParseArgs(Configuration * cf, int argc, char **argv);

vector<string> tokenize_str(string const & data);
vector<int> tokenize_int(string const & data);
vector<double> tokenize_float(string const & data);

#endif
