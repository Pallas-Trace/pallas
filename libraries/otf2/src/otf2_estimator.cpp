/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2014,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2014,
 * Technische Universitaet Dresden, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#include <cstring>
#include <iostream>
#include <string>


#include "otf2/OTF2_EventSizeEstimator.h"
#include "otf2/OTF2_GeneralDefinitions.h"
#define OTF2_TOOL_NAME "otf2-estimator"
#include "otf2/otf2_tools.hpp"

using namespace std;


static string
get_token( const string& input, size_t& pos )
{
  size_t end_pos = input.find_first_not_of( " \t", pos );
  if ( end_pos == string::npos )
  {
    pos = string::npos;
    return "";
  }

  pos = input.find_first_of( " \t", end_pos );
  return input.substr( end_pos, pos - end_pos );
}

#include "otf2_estimator_inc.cpp"

int
main( int argc, char** argv )
{
  if ( argc > 1 )
  {
    if ( strcmp( argv[ 1 ], "--help" ) == 0 || strcmp( argv[ 1 ], "-h" ) == 0 )
    {
      string usage = "Usage: otf2-estimator [OPTION]...\n"
        "This tool estimates the size of OTF2 events.\n"
        "It will open a prompt to type in commands.\n"
        "\n"
        "Options:\n"
        "  -V, --version       Print version information.\n"
        "  -h, --help          Print this help information.\n"
        "\n"
        "Commands:\n"
        "   list definitions              Lists all known definition names.\n"
        "   list events                   Lists all known event names.\n"
        "   list types                    Lists all known type names.\n"
        "   set <DEFINITION> <NUMBER>     Specifies the number of definitions of a\n"
        "                                 type of definitions.\n"
        "   get DefChunkSize              Prints the estimated definition chunk size.\n"
        "   get Timestamp                 Prints the size of a timestamp.\n"
        "   get AttributeList [TYPES...]  Prints the estimated size for an attribute\n"
        "                                 list with the given number of entries and\n"
        "                                 types.\n"
        "   get <EVENT> [ARGS...]         Prints the estimated size of records for\n"
        "                                 <EVENT>.\n"
        "   exit                          Exits the tool.\n"
        "\n"
        "This tool provides a command line interface to the estimator API of the OTF2\n"
        "library. It is based on a stream based protocol.  Commands are send to the\n"
        "standard input stream of the program and the result is written to the standard\n"
        "output stream of the program.  All definition and event names are in there\n"
        "canonical CamelCase form.  Numbers are printed in decimal.  The TYPES are in\n"
        "ALL_CAPS.  See the output of the appropriate 'list' commands.  Arguments are\n"
        "separated with an arbitrary number of white space.  The 'get' commands are using\n"
        "everything after the first white space separator verbatim as a key, which is\n"
        "then printed in the output line and appended with the estimated size.\n"
        "\n"
        "Here is a simple example.  We have at most 4 region definitions and one metric\n"
        "definition.  We want to know the size of a timestamp, enter, and leave event,\n"
        "and a metric event with 4 values.\n"
        "\n"
        "cat <<EOC | otf2-estimator\n"
        "set Region 4\n"
        "set Metric 1\n"
        "get Timestamp\n"
        "get Enter\n"
        "get Leave\n"
        "get Metric  4\n"
        "exit\n"
        "EOC\n"
        "Timestamp 9\n"
        "Enter 3\n"
        "Leave 3\n"
        "Metric  4 44"
        ;
      cout << usage.c_str() << endl;
      cout << "Report bugs to <catherine.guelque@telecom-sudparis.eu>" << endl;
      exit( EXIT_SUCCESS );
    }
    else if ( strcmp( argv[ 1 ], "--version" ) == 0 || strcmp( argv[ 1 ], "-V" ) == 0 )
    {
      out( "version " << OTF2_VERSION );
      exit( EXIT_SUCCESS );
    }
    else
    {
      die( "unrecognized parameter: '" << argv[ 1 ] << "'" );
    }
  }

  OTF2_EventSizeEstimator* estimator = OTF2_EventSizeEstimator_New();

  while ( true )
  {
    if ( cin.eof() )
    {
      break;
    }

    string input;
    getline( cin, input );
    size_t pos           = 0;
    string current_token = get_token( input, pos );

    if ( current_token == "exit" )
    {
      break;
    }
    else if ( current_token == "get" )
    {
      handle_get( estimator, input, pos );
    }
    else if ( current_token == "set" )
    {
      handle_set( estimator, input, pos );
    }
    else if ( current_token == "list" )
    {
      current_token = get_token( input, pos );
      if ( current_token == "events" )
      {
        print_event_list();
      }
      else if ( current_token == "definitions" )
      {
        print_definition_list();
      }
      else if ( current_token == "types" )
      {
        print_type_list();
      }
      else
      {
        die( "can not list: '" << current_token << "'" );
      }
    }
    else if ( current_token != "" )
    {
      die( "unknown command: '" << current_token << "'" );
    }
  }

  OTF2_EventSizeEstimator_Delete( estimator );
  return EXIT_SUCCESS;
}
