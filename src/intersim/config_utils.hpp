#ifndef _CONFIG_UTILS_HPP_
#define _CONFIG_UTILS_HPP_

#include<stdio.h>
#include<string>
#include<map>

extern int configparse( );

class Configuration {
   static Configuration *theConfig;
   FILE *_config_file;

protected:
   map<string,char *>       _str_map;
   map<string,unsigned int> _int_map;
   map<string,double>       _float_map;

public:
   Configuration( );

   void AddStrField( const string &field, const string &value );

   void Assign( const string &field, const string &value );
   void Assign( const string &field, unsigned int value );
   void Assign( const string &field, double value );

   void GetStr( const string &field, string &value, const string &def = "" ) const;
   unsigned int GetInt( const string &field, unsigned int def = 0 ) const;
   double GetFloat( const string &field, double def = 0.0 ) const;

   void Parse( const string& filename );
   void Parse( const char* filename );

   int  Input( char *line, int max_size );
   void ParseError( const string &msg, unsigned int lineno ) const;

   static Configuration *GetTheConfig( );
};

bool ParseArgs( Configuration *cf, int argc, char **argv );

#endif


