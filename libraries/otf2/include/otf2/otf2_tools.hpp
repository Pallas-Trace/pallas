/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013,
 * Technische Universitaet Dresden, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 */

/** @internal
 *
 *  @file
 *
 *  @brief      Helper class for OTF2 tools.
 */

#ifndef OTF2_TOOLS_HPP
#define OTF2_TOOLS_HPP

#ifndef OTF2_TOOL_NAME
#error No OTF2_TOOL_NAME given
#endif

/* ___ Output handling. _____________________________________________________ */

#define out( msg ) \
    do { \
        std::cout << OTF2_TOOL_NAME << ": " << msg << std::endl; \
    } while ( 0 )

#define err( msg ) \
    do { \
        std::cerr << OTF2_TOOL_NAME << ": " << msg << std::endl; \
    } while ( 0 )

#define warn( msg ) \
    do { \
        std::cerr << OTF2_TOOL_NAME << ": warning: " << msg << std::endl; \
    } while ( 0 )

#define die( msg ) \
    do { \
        err( msg ); \
        std::cerr << "Try '" << OTF2_TOOL_NAME << " --help' for more information." << std::endl; \
        exit( EXIT_FAILURE ); \
    } while ( 0 )

#endif // OTF2_TOOLS_HPP
