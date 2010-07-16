#include "booksim.hpp"
#include <iostream>
#include <stdlib.h>
#include <cstring>

#include "config_utils.hpp"

Configuration *Configuration::theConfig = 0;

Configuration::Configuration( )
{
   theConfig = this;
   _config_file = 0;
}

void Configuration::AddStrField( const string &field, const string &value )
{
   _str_map[field] = strdup( value.c_str( ) );
}

void Configuration::Assign( const string &field, const string &value )
{
   map<string,char *>::const_iterator match;

   match = _str_map.find( field );
   if ( match != _str_map.end( ) ) {
      free( _str_map[field] );
      _str_map[field] = strdup( value.c_str( ) );
   } else {
      string errmsg = "Unknown field ";
      errmsg += field;

      ParseError( errmsg, 0 );
   }
}

void Configuration::Assign( const string &field, unsigned int value )
{
   map<string,unsigned int>::const_iterator match;

   match = _int_map.find( field );
   if ( match != _int_map.end( ) ) {
      _int_map[field] = value;
   } else {
      string errmsg = "Unknown field ";
      errmsg += field;

      ParseError( errmsg, 0 );
   }
}

void Configuration::Assign( const string &field, double value )
{
   map<string,double>::const_iterator match;

   match = _float_map.find( field );
   if ( match != _float_map.end( ) ) {
      _float_map[field] = value;
   } else {
      string errmsg = "Unknown field ";
      errmsg += field;

      ParseError( errmsg, 0 );
   }
}

void Configuration::GetStr( const string &field, string &value, const string &def ) const
{
   map<string,char *>::const_iterator match;

   match = _str_map.find( field );
   if ( match != _str_map.end( ) ) {
      value = match->second;
   } else {
      value = def;
   }
}

unsigned int Configuration::GetInt( const string &field, unsigned int def ) const
{
   map<string,unsigned int>::const_iterator match;
   unsigned int r = def;

   match = _int_map.find( field );
   if ( match != _int_map.end( ) ) {
      r = match->second;
   }

   return r;
}

double Configuration::GetFloat( const string &field, double def ) const
{  
   map<string,double>::const_iterator match;
   double r = def;

   match = _float_map.find( field );
   if ( match != _float_map.end( ) ) {
      r = match->second;
   }

   return r;
}

void Configuration::Parse( const string& filename )
{
   if ( ( _config_file = fopen( filename.c_str( ), "r" ) ) == 0 ) {
      cerr << "Could not open configuration file " << filename << endl;
      exit( -1 );
   }

   configparse( );

   fclose( _config_file );
   _config_file = 0;
}

void Configuration::Parse( const char* filename )
{
   if ( ( _config_file = fopen( filename , "r" ) ) == 0 ) {
      cerr << "Could not open configuration file " << filename << endl;
      exit( -1 );
   }

   configparse( );

   fclose( _config_file );
   _config_file = 0;
}


int Configuration::Input( char *line, int max_size )
{
   int length = 0;

   if ( _config_file ) {
      length = fread( line, 1, max_size, _config_file );
   }

   return length;
}

void Configuration::ParseError( const string &msg, unsigned int lineno ) const
{
   if ( lineno ) {
      cerr << "Parse error on line " << lineno << " : " << msg << endl;
   } else {
      cerr << "Parse error : " << msg << endl;
   }

   exit( -1 );
}

Configuration *Configuration::GetTheConfig( )
{
   return theConfig;
}

//============================================================

int config_input( char *line, int max_size )
{
   return Configuration::GetTheConfig( )->Input( line, max_size );
}

bool ParseArgs( Configuration *cf, int argc, char **argv )
{
   bool rc = false;

   if ( argc > 1 ) {
      cf->Parse( argv[1] );
      rc = true;
   }

   return rc;
}
