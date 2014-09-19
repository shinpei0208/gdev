/*!
	\file StatisticDatabase.cpp
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Saturday March 28, 2009
	
	\brief The source file for the StatisticDatabase class
*/

#ifndef STATISTIC_DATABSE_CPP_INCLUDED
#define STATISTIC_DATABSE_CPP_INCLUDED

#include <hydrazine/interface/StatisticDatabase.h>
#include <hydrazine/interface/macros.h>
#include <hydrazine/interface/debug.h>
#include <cassert>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace hydrazine
{

	StatisticDatabase::Statistic::Statistic( Type _type, Id _id ) : 
		type( _type ), id( _id )
	{
	
	}

	StatisticDatabase::Statistic::~Statistic()
	{
	
	}
	
	void StatisticDatabase::Statistic::aggregate( const Statistic& )
	{
	
	}
	
	hydrazine::StatisticDatabase::Statistic*
		StatisticDatabase::Statistic::clone( bool copy ) const
	{
	
		return new Statistic( type, id );
	
	}
	
	std::string	StatisticDatabase::Statistic::toString() const
	{
		return "";
	}
	
	StatisticDatabase::Generator::Generator( StatisticDatabase& database ) : 
		_database( database ), _databaseRegistered( &database != 0 )
	{
	
	}
	
	StatisticDatabase::IdMap StatisticDatabase::_emptyMap;
	
	StatisticDatabase::StatisticDatabase()
	{
	
		_size = 0;
		_aggregate = new Statistic( STATISTIC_INVALID_TYPE, 
			STATISTIC_INVALID_ID );
	
	}
	
	StatisticDatabase::~StatisticDatabase()
	{
	
		for( StatisticMap::iterator statistic = _map.begin(); 
			statistic != _map.end(); ++statistic )
		{		
			for( IdMap::iterator fi = statistic->second.begin(); 
				fi != statistic->second.end(); ++fi )
			{			
				delete fi->second;			
			}		
		}
		
		delete _aggregate;	
	}
	
	void StatisticDatabase::insert( Statistic* statistic )
	{	
		StatisticMap::iterator map = _map.find( statistic->type );
		
		if( map == _map.end() )
		{			
			IdMap newMap;
			map = _map.insert( std::make_pair( statistic->type, 
				newMap ) ).first;					
		}
		
		assert( map->second.count( statistic->id ) == 0 );
		
		map->second.insert( std::make_pair( statistic->id, 
			statistic->clone( true ) ) );
		++_size;
	}
	
	const StatisticDatabase::IdMap& 
		StatisticDatabase::find( Statistic::Type type )
	{	
		StatisticMap::iterator map = _map.find( type );
		
		if( map == _map.end() )
		{		
			return _emptyMap;		
		}
		
		return map->second;	
	}
	
	StatisticDatabase::Statistic* StatisticDatabase::find( Statistic::Type type,
		Statistic::Id id )
	{	
		StatisticMap::iterator map = _map.find( type );
		
		if( map == _map.end() )
		{		
			return 0;		
		}
		
		IdMap::iterator statistic = map->second.find( id );
		
		if( statistic == map->second.end() )
		{		
			return 0;		
		}
		
		report( "Looked up statistic : " << statistic->second->toString() );
		
		return statistic->second;	
	}

	void StatisticDatabase::erase( Statistic* statistic )
	{	
		StatisticMap::iterator map = _map.find( statistic->type );
		assert( map != _map.end() );
		
		IdMap::iterator fi = map->second.find( statistic->id );
		assert( fi != map->second.end() );
		
		map->second.erase( fi );
		
		if( map->second.empty() )
		{		
			_map.erase( map );		
		}
		
		--_size;	
	}

	const StatisticDatabase::Statistic* 
		StatisticDatabase::aggregate( Statistic::Type type )
	{	
		delete _aggregate;
		
		StatisticMap::iterator map = _map.find( type );
		assert( map != _map.end() );
		assert( !map->second.empty() );
		
		IdMap::iterator statistic = map->second.begin();
		
		_aggregate = static_cast< Statistic* >( ( 
			statistic->second )->clone( true ) );
		
		for( ; statistic != map->second.end(); ++statistic )
		{		
			_aggregate->aggregate( *statistic->second );		
		}
		
		return _aggregate;	
	}
	
	bool StatisticDatabase::empty() const
	{	
		return _size == 0;	
	}

	unsigned int StatisticDatabase::size() const
	{	
		return _size;	
	}
	
	void StatisticDatabase::clear()
	{	
		for( StatisticMap::iterator statistic = _map.begin(); 
			statistic != _map.end(); ++statistic )
		{		
			for( IdMap::iterator fi = statistic->second.begin(); 
				fi != statistic->second.end(); ++fi )
			{			
				delete fi->second;			
			}		
		}
		
		_size = 0;	
	}
	
	std::string StatisticDatabase::toString() const
	{	
		std::string result;
	
		for( StatisticMap::const_iterator statistic = _map.begin(); 
			statistic != _map.end(); ++statistic )
		{		
			for( IdMap::const_iterator fi = statistic->second.begin(); 
				fi != statistic->second.end(); ++fi )
			{			
				result += fi->second->toString() + "\n\n";			
			}		
		}
		
		return result;	
	}

}

#endif

