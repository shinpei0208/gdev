/*!
	\file StatisticDatabase.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Saturday March 28, 2009
	\brief The header file for the StatisticDatabase class
*/

#ifndef STATISTIC_DATABSE_H_INCLUDED
#define STATISTIC_DATABSE_H_INCLUDED

#include <hydrazine/interface/SystemCompatibility.h>

#define STATISTIC_INVALID_TYPE 0xffffffff
#define STATISTIC_INVALID_ID 0xffffffff

namespace hydrazine
{

	/*!	
		\brief A collection of different type of statistics to support 
			aggregation across different types.	
	*/
	class StatisticDatabase
	{
	
		public:		
			/*!	\brief A collection of information of a given type */
			class Statistic
			{			
				public:				
					/*!	\brief A unique type for a derived class of Statistic */
					typedef unsigned int Type;
					
					/*!	\brief A unique Id within a type for a Statistic */
					typedef unsigned int Id;
				
				public:				
					/*!	\brief The type of this Statistic.
						
						Statistics of the same type can be aggregated.					
					*/
					const Type type;
					
					/*!	\brief The Id of the statistic */
					const Id id;
					
				public:
				
					/*!	\brief Constructor for setting the type	*/
					Statistic( Type type, Id id );
					
					/*!	\brief Virtual destructor */
					virtual ~Statistic();
					
					/*!	\brief Aggregate the values in this statistic with
							another of the same type.				
					*/
					virtual void aggregate( const Statistic& statistic );
					
				public:
					virtual Statistic* clone( bool copy = true ) const; 
					std::string toString() const;
			
			};
			
			/*!
				\brief A class that generates Statistics for a specific databse
			*/
			class Generator
			{			
				protected:				
					/*!	\brief Reference to the connected StatisticDatabase	*/
					StatisticDatabase& _database;
					
					/*!	\brief Set to true if the StatisticDatabase	is valid */
					const bool _databaseRegistered;
					
				public:				
					/*!	\brief The constructor registers the class with a 
							database
							
						\param database A reference to the database being 
							registered. Should be a null pointer if invalid.
					*/
					Generator( StatisticDatabase& database );
					
			};

		private:		
			typedef std::unordered_map< Statistic::Id, Statistic* > IdMap;
			typedef std::unordered_map< Statistic::Type, IdMap > StatisticMap;

		private:
			static IdMap _emptyMap;

		private:		
			Statistic* _aggregate;
			StatisticMap _map;
			unsigned int _size;
	
		public:		
			/*!	\brief Constructor */
			StatisticDatabase();

			/*!	\brief Destructor */
			~StatisticDatabase();
			
			/*!	\brief Insert a new statistic into the database.
				
				\param statistic The Statistic being observed.
			*/
			void insert( Statistic* statistic );

			/*!	\brief Find a statistic in the database
				
				\param type The type of statistic being looked up
				\return Reference to a map of statistics with this type
			*/
			const IdMap& find( Statistic::Type type );

			/*!	\brief Find a statistic in the database
				
				\param type The Type of statistic being looked up
				\param id The Id of Statistic being looked up
				\return Pointer to the Statistic with this type and id
				
				This will be zero if the statistic does not exist
				
			*/
			Statistic* find( Statistic::Type type, Statistic::Id id );
			
			/*!	\brief Erase a single statistic
				
				\param statistic A pointer to the statistic being erased.
			*/
			void erase( Statistic* statistic );
			
			/*!	\brief Aggregate all of the statistics contained of a specific
					type.  
					
				This returns a locally owned object.  It is only valid until
				the next call to aggregate.
				
				\param type The statistic type to aggregate over.
				\return A pointer to the newly aggregated Statistic
			*/
			const Statistic* aggregate( Statistic::Type type );
			
			/*!	\brief Is the database empty
				\return True if empty
			*/
			bool empty() const;
			
			/*!	\brief Get the number of Statistics in the database
				\return The number of Statistics
			*/
			unsigned int size() const;
			
			/*!	\brief Clear all Statistics from the database */
			void clear();
	
		public:
		
			/*! \brief Create a string representation of all statistics	*/
			std::string toString() const;
	
	};

}

#endif

