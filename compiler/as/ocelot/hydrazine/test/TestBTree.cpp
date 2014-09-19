/*!
	\file TestBTree.cpp
	\date Monday May 18, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the TestBTree class.
*/

#ifndef TEST_B_TREE_CPP_INCLUDED
#define TEST_B_TREE_CPP_INCLUDED

#include <hydrazine/test/TestBTree.h>
#include <hydrazine/implementation/Exception.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <fstream>

namespace test
{

	void TestBTree::_init( Vector& v )
	{
		for( Vector::iterator fi = v.begin(); fi != v.end(); ++fi )
		{
			*fi = random() % elements;
		}
	}
	
	static void dumpTree( TestBTree::Tree& tree, const std::string& path, 
		const std::string& message = "failed" )
	{
		std::stringstream stream;
		stream << path << "/tree_" + message + ".dot";
		std::ofstream out( stream.str().c_str() );
		if( !out.is_open() )
		{
			throw hydrazine::Exception( 
				"Could not open file " + stream.str() );
		}
		out << tree;
	}

	bool TestBTree::testRandom()
	{
		typedef std::pair< Map::iterator, bool > MapInsertion;
		typedef std::pair< Tree::iterator, bool > TreeInsertion;

		status << "Running Test Random\n";

		Map map;
		Tree tree;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 3 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					MapInsertion mapInsertion = map.insert( 
						std::make_pair( vector[ index ], i ) );
					TreeInsertion treeInsertion = tree.insert( 
						std::make_pair( vector[ index ], i ) );
					if( mapInsertion.second != treeInsertion.second )
					{
						status << "At index " << i << std::boolalpha 
							<< " map insertion " 
							<< mapInsertion.second 
							<< " did not match tree insertion " 
							<< treeInsertion.second << "\n";
						dumpTree( tree, path );
						return false;
					}
					
					if( mapInsertion.first->first 
						!= treeInsertion.first->first )
					{
						status << "At index " << i << " map key " 
							<< mapInsertion.first->first 
							<< " did not match tree key " 
							<< treeInsertion.first->first << "\n";
						dumpTree( tree, path );
						return false;
					}

					if( mapInsertion.first->second 
						!= treeInsertion.first->second )
					{
						status << "At index " << i << " map value " 
							<< mapInsertion.first->second 
							<< " did not match tree value " 
							<< treeInsertion.first->second << "\n";
						dumpTree( tree, path );
						return false;
					}
					
					break;
				}
				
				case 1:
				{
					size_t index = random() % vector.size();
					Map::const_iterator mi = map.find( vector[ index ] );
					Tree::const_iterator ti = tree.find( vector[ index ] );
					bool mapEnd = mi == map.end();
					bool treeEnd = ti == tree.end();
										
					if( mapEnd || treeEnd )
					{
						if( mapEnd != treeEnd )
						{
							status << "At index " << i << std::boolalpha 
								<< " for key " << vector[ index ] << " map end "
								<< mapEnd << " did not match tree end " 
								<< treeEnd << "\n";
							dumpTree( tree, path );
							return false;						
						}
					}
					else
					{
						if( mi->first != ti->first )
						{
							status << "At index " << i << " map key " 
								<< mi->first << " did not match tree key " 
								<< ti->first << "\n";
							dumpTree( tree, path );
							return false;
						}

						if( mi->second != ti->second )
						{
							status << "At index " << i << " map value " 
								<< mi->second << " did not match tree value " 
								<< ti->second << "\n";
							dumpTree( tree, path );
							return false;
						}
					}
				}
				
				case 2:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					bool mapEnd = mi == map.end();
					bool treeEnd = ti == tree.end();
										
					if( mapEnd || treeEnd )
					{
						if( mapEnd != treeEnd )
						{
							status << "At index " << i << std::boolalpha 
								<< " for key " << vector[ index ] << " map end "
								<< mapEnd << " did not match tree end " 
								<< treeEnd << "\n";
							dumpTree( tree, path );
							return false;	
						}
					}
					else
					{
						map.erase( mi );
						tree.erase( ti );
					}
				}
			}
		}
		
		status << "Test Random Passed\n";
		return true;
	}

	bool TestBTree::testClear()
	{
		status << "Running Test Clear\n";

		Map map;
		Tree tree;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					tree.insert( std::make_pair( vector[ index ], i ) );
					
					if( map.size() != tree.size() )
					{
						status << "Map size " << map.size() 
							<< " does not match tree size " 
							<< tree.size() << "\n";
						dumpTree( tree, path );
						return false;
					}
					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
						
						if( map.size() != tree.size() )
						{
							status << "Map size " << map.size() 
								<< " does not match tree size " 
								<< tree.size() << "\n";
							dumpTree( tree, path );
							return false;
						}
					}
					break;
				}
			}
		}
		
		map.clear();
		tree.clear();
		
		if( map.size() != tree.size() )
		{
			status << "Ater clear, map size " << map.size() 
				<< " does not match tree size " 
				<< tree.size() << "\n";
			dumpTree( tree, path );
			return false;
		}
		
		if( !tree.empty() )
		{
			status << "Ater clear, tree reported not empty\n";
			dumpTree( tree, path );
			return false;
		}
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					tree.insert( std::make_pair( vector[ index ], i ) );
					
					if( map.size() != tree.size() )
					{
						status << "Map size " << map.size() 
							<< " does not match tree size " 
							<< tree.size() << "\n";
						dumpTree( tree, path );
						return false;
					}
					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
						
						if( map.size() != tree.size() )
						{
							status << "Map size " << map.size() 
								<< " does not match tree size " 
								<< tree.size() << "\n";
							dumpTree( tree, path );
							return false;
						}
					}
					break;
				}
			}
		}
	
		status << "Test Clear Passed\n";
		return true;
	}
	
	bool TestBTree::testIteration()
	{
		status << "Running Test Forward Iteration\n";

		Map map;
		Tree tree;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					tree.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
					}
					break;
				}
			}
		}
		
		{
			Map::iterator mi = map.begin();
			Tree::iterator ti = tree.begin();
	
			for( ; mi != map.end() && ti != tree.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Forward iteration failed, map key " << mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( tree, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Forward iteration failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second << "\n";
					dumpTree( tree, path );
					return false;
				}
			}
			
			if( mi != map.end() )
			{
				status << "Forward iteration failed, map did not hit the end\n";
				dumpTree( tree, path );
				return false;
			}
		
			if( ti != tree.end() )
			{
				status << "Forward iteration failed, tree did not hit end\n";
				dumpTree( tree, path );
				return false;
			}		
		}
		
		{
			Map::const_iterator mi = map.begin();
			Tree::const_iterator ti = tree.begin();
	
			for( ; mi != map.end() && ti != tree.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Forward const iteration failed, map key " 
						<< mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( tree, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Forward const iteration failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second << "\n";
					dumpTree( tree, path );
					return false;
				}
			}
			
			if( mi != map.end() )
			{
				status << "Forward const iteration failed, map did " 
					<< "not hit the end\n";
				dumpTree( tree, path );
				return false;
			}
		
			if( ti != tree.end() )
			{
				status << "Forward const iteration failed, tree" 
					<< " did not hit end\n";
				dumpTree( tree, path );
				return false;
			}		
		}
		
		{
			Map::reverse_iterator mi = map.rbegin();
			Tree::reverse_iterator ti = tree.rbegin();
	
			for( ; mi != map.rend() && ti != tree.rend(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Reverse iteration failed, map key " 
						<< mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( tree, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Reverse iteration failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second << "\n";
					dumpTree( tree, path );
					return false;
				}
			}
			
			if( mi != map.rend() )
			{
				status << "Reverse iteration failed, map did " 
					<< "not hit the end\n";
				dumpTree( tree, path );
				return false;
			}
		
			if( ti != tree.rend() )
			{
				status << "Reverse iteration failed, tree" 
					<< " did not hit end\n";
				dumpTree( tree, path );
				return false;
			}		
		}
		
		{
			Map::const_reverse_iterator mi = map.rbegin();
			Tree::const_reverse_iterator ti = tree.rbegin();
	
			for( ; mi != map.rend() && ti != tree.rend(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Reverse const iteration failed, map key " 
						<< mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( tree, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Reverse const iteration failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second << "\n";
					dumpTree( tree, path );
					return false;
				}
			}
			
			if( mi != map.rend() )
			{
				status << "Reverse const iteration failed, map did " 
					<< "not hit the end\n";
				dumpTree( tree, path );
				return false;
			}
		
			if( ti != tree.rend() )
			{
				status << "Reverse const iteration failed, tree" 
					<< " did not hit end\n";
				dumpTree( tree, path );
				return false;
			}		
		}
		
		status << "Test Iteration Passed\n";
		return true;
			
	}
	
	bool TestBTree::testComparisons()
	{
		status << "Running Test Comparisons\n";

		Map map, map1;
		Tree tree, tree1;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					tree.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
					}
					break;
				}
			}
		}
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map1.insert( std::make_pair( vector[ index ], i ) );
					tree1.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map1.find( vector[ index ] );
					Tree::iterator ti = tree1.find( vector[ index ] );
					
					if( mi != map1.end() )
					{
						map1.erase( mi );
						tree1.erase( ti );
					}
					break;
				}
			}
		}
		
		bool mapResult;
		bool treeResult;
		
		mapResult = map == map1;
		treeResult = tree == tree1;
		
		if( mapResult != treeResult )
		{
			status << "For comparison ==, map result " << mapResult
				<< " does not match tree result " << treeResult << "\n";
			dumpTree( tree, path, "failed" );
			dumpTree( tree1, path, "failed1" );
			return false;
		}

		mapResult = map != map1;
		treeResult = tree != tree1;

		if( mapResult != treeResult )
		{
			status << "For comparison !=, map result " << mapResult
				<< " does not match tree result " << treeResult << "\n";
			dumpTree( tree, path, "failed" );
			dumpTree( tree1, path, "failed1" );
			return false;
		}

		mapResult = map < map1;
		treeResult = tree < tree1;

		if( mapResult != treeResult )
		{
			status << "For comparison <, map result " << mapResult
				<< " does not match tree result " << treeResult << "\n";
			dumpTree( tree, path, "failed" );
			dumpTree( tree1, path, "failed1" );
			return false;
		}

		mapResult = map <= map1;
		treeResult = tree <= tree1;

		if( mapResult != treeResult )
		{
			status << "For comparison <=, map result " << mapResult
				<< " does not match tree result " << treeResult << "\n";
			dumpTree( tree, path, "failed" );
			dumpTree( tree1, path, "failed1" );
			return false;
		}

		mapResult = map > map1;
		treeResult = tree > tree1;

		if( mapResult != treeResult )
		{
			status << "For comparison >, map result " << mapResult
				<< " does not match tree result " << treeResult << "\n";
			dumpTree( tree, path, "failed" );
			dumpTree( tree1, path, "failed1" );
			return false;
		}

		mapResult = map >= map1;
		treeResult = tree >= tree1;

		if( mapResult != treeResult )
		{
			status << "For comparison >=, map result " << mapResult
				<< " does not match tree result " << treeResult << "\n";
			dumpTree( tree, path, "failed" );
			dumpTree( tree1, path, "failed1" );
			return false;
		}
	
		status << "Test Comparison Passed\n";
		return true;
		
	}

	bool TestBTree::testSearching()
	{
		status << "Running Test Searching\n";

		Map map, map1;
		Tree tree, tree1;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 3 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					tree.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
					}
					break;
				}
				
				case 2:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						if( ti == tree.end() )
						{
							status << "Map found key " << mi->first
								<< ", but it was not found in the tree.\n";
							dumpTree( tree, path );
							return false;
						}
						if( mi->first != ti->first )
						{
							status << "Find failed, map key " << mi->first
								<< " does not match tree key " << ti->first 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
						if( mi->second != ti->second )
						{
							status << "Find failed, map value " << mi->second
								<< " does not match tree value " << ti->second 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
					}
					
					mi = map.lower_bound( vector[ index ] );
					ti = tree.lower_bound( vector[ index ] );
					
					if( mi != map.end() )
					{
						if( ti == tree.end() )
						{
							status << "Map lower_bound found key " << mi->first
								<< ", but it was not found in the tree.\n";
							dumpTree( tree, path );
							return false;
						}
						if( mi->first != ti->first )
						{
							status << "Lower_bound failed, map key " 
								<< mi->first
								<< " does not match tree key " << ti->first 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
						if( mi->second != ti->second )
						{
							status << "Lower_bound failed, map value " 
								<< mi->second
								<< " does not match tree value " << ti->second 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
					}
					
					mi = map.upper_bound( vector[ index ] );
					ti = tree.upper_bound( vector[ index ] );
					
					if( mi != map.end() )
					{
						if( ti == tree.end() )
						{
							status << "Map upper_bound found key " << mi->first
								<< ", but it was not found in the tree.\n";
							dumpTree( tree, path );
							return false;
						}
						if( mi->first != ti->first )
						{
							status << "Upper_bound failed, map key " 
								<< mi->first
								<< " does not match tree key " << ti->first 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
						if( mi->second != ti->second )
						{
							status << "Upper_bound failed, map value " 
								<< mi->second
								<< " does not match tree value " << ti->second 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
					}
					
					std::pair< Map::iterator, Map::iterator > 
						mp = map.equal_range( vector[ index ] );
					std::pair< Tree::iterator, Tree::iterator > 
						tp = tree.equal_range( vector[ index ] );
					
					if( mp.first != map.end() )
					{
						if( tp.first == tree.end() )
						{
							status << "Map equal range lower key " 
								<< mp.first->first
								<< ", but it was not found in the tree.\n";
							dumpTree( tree, path );
							return false;
						}
						if( mp.first->first != tp.first->first )
						{
							status << "Equal_range failed, lower map key " 
								<< mp.first->first
								<< " does not match tree key " 
								<< tp.first->first 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
						if( mp.first->second != tp.first->second )
						{
							status << "Equal_range failed, lower map value " 
								<< mp.first->second
								<< " does not match tree value " 
								<< tp.first->second << "\n";
							dumpTree( tree, path );
							return false;
						}
					}
					
					if( mp.second != map.end() )
					{
						if( tp.second == tree.end() )
						{			dumpTree( tree, path, "failed" );

							status << "Map equal range upper key " 
								<< mp.second->first
								<< ", but it was not found in the tree.\n";
							dumpTree( tree, path );
							return false;
						}
						if( mp.second->first != tp.second->first )
						{
							status << "Equal_range failed, upper map key " 
								<< mp.second->first
								<< " does not match tree key " 
								<< tp.second->first 
								<< "\n";
							dumpTree( tree, path );
							return false;
						}
						if( mp.second->second != tp.second->second )
						{
							status << "Equal_range failed, upper map value " 
								<< mp.second->second
								<< " does not match tree value " 
								<< tp.second->second << "\n";
							dumpTree( tree, path );
							return false;
						}
					}
					
					break;
				}
			}
		}
		
		status << "Test Searching Passed.\n";
		return true;
	}

	bool TestBTree::testSwap()
	{
		status << "Running Test Swap\n";

		Map map, map1;
		Tree tree, tree1;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					tree.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
					}
					break;
				}
			}
		}
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map1.insert( std::make_pair( vector[ index ], i ) );
					tree1.insert( std::make_pair( vector[ index ], i ) );					
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map1.find( vector[ index ] );
					Tree::iterator ti = tree1.find( vector[ index ] );
					
					if( mi != map1.end() )
					{
						map1.erase( mi );
						tree1.erase( ti );
					}
					break;
				}
			}
		}
		
		map.swap( map1 );
		tree.swap( tree1 );
		
		{
			Map::iterator mi = map.begin();
			Tree::iterator ti = tree.begin();

			for( ; mi != map.end() && ti != tree.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Swap failed, map key " << mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( tree, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Swap failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second 
						<< "\n";
					dumpTree( tree, path );
					return false;
				}
			}
			
		}
		
		{
			Map::iterator mi = map1.begin();
			Tree::iterator ti = tree1.begin();

			for( ; mi != map1.end() && ti != tree1.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Swap failed, map key " << mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( tree1, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Swap failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second 
						<< "\n";
					dumpTree( tree1, path );
					return false;
				}
			}
		}
		
		status << "  Test Swap Passed.\n";
		return true;		
	}
	
	bool TestBTree::testInsert()
	{
		status << "Running Test Insert\n";

		Map map;
		Tree tree;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					Tree::iterator fi = 
						tree.lower_bound( vector[ index ] );
					fi = tree.insert( fi, 
						std::make_pair( vector[ index ], i ) );
					if( fi->first != vector[ index ] )
					{
						status << "Insert failed, returned iterator with key " 
						<< fi->first << " does not match inserted value " 
						<< vector[ index ] << "\n";
						dumpTree( tree, path );
						return false;
					}		
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
					}
					break;
				}
			}
		}
		
		tree.clear();
		tree.insert( map.begin(), map.end() );
		
		{
			Map::iterator mi = map.begin();
			Tree::iterator ti = tree.begin();

			for( ; mi != map.end() && ti != tree.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Insert failed, map key " << mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( tree, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Insert failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second 
						<< "\n";
					dumpTree( tree, path );
					return false;
				}
			}
		}
		
		status << "  Test Insert Passed.\n";
		return true;
	}

	bool TestBTree::testErase()
	{
		status << "Running Test Insert\n";

		Map map;
		Tree tree;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					Tree::iterator fi = 
						tree.lower_bound( vector[ index ] );
					fi = tree.insert( fi, 
						std::make_pair( vector[ index ], i ) );
					if( fi->first != vector[ index ] )
					{
						status << "Insert failed, returned iterator with key " 
						<< fi->first << " does not match inserted value " 
						<< vector[ index ] << "\n";
						dumpTree( tree, path );
						return false;
					}		
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					size_t index1 = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Map::iterator mi1 = map.find( vector[ index1 ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					Tree::iterator ti1 = tree.find( vector[ index1 ] );
					
					if( mi == map.end() && mi1 == map.end() )
					{
						break;
					}
					
					if( mi == map.end() && mi1 != map.end() )
					{
						std::swap( mi, mi1 );
						std::swap( ti, ti1 );
					}
					else if( mi != map.end() && mi1 != map.end() )
					{
						if( mi1->first < mi->first )
						{
							std::swap( mi, mi1 );
							std::swap( ti, ti1 );
						}
					}
					
					map.erase( mi, mi1 );
					tree.erase( ti, ti1 );
					break;
				}
			}
		}
			
		{
			Map::iterator mi = map.begin();
			Tree::iterator ti = tree.begin();

			for( ; mi != map.end() && ti != tree.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Erase failed, map key " << mi->first
						<< " does not match tree key " << ti->first 
						<< "\n";
					dumpTree( tree, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Erase failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second 
						<< "\n";
					dumpTree( tree, path );
					return false;
				}
			}
		}
		
		status << "  Test Erase Passed.\n";
		return true;
	}

	bool TestBTree::testCopy()
	{
		status << "Running Test Copy\n";

		Map map;
		Tree tree;
		Vector vector( elements );
		_init( vector );
		
		for( unsigned int i = 0; i < iterations; ++i )
		{
			switch( random() % 2 )
			{
				case 0:
				{
					size_t index = random() % vector.size();
					map.insert( std::make_pair( vector[ index ], i ) );
					tree.insert( std::make_pair( vector[ index ], i ) );						
					break;
				}
								
				case 1:
				{
					size_t index = random() % vector.size();
					Map::iterator mi = map.find( vector[ index ] );
					Tree::iterator ti = tree.find( vector[ index ] );
					
					if( mi != map.end() )
					{
						map.erase( mi );
						tree.erase( ti );
					}
					break;
				}
			}
		}
		
		Tree copy( tree );
		
		{
			Map::iterator mi = map.begin();
			Tree::iterator ti = copy.begin();

			for( ; mi != map.end() && ti != copy.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Copy failed, map key " << mi->first
						<< " does not match tree key " << ti->first 
						<< "\n";
					dumpTree( copy, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Copy failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second 
						<< "\n";
					dumpTree( copy, path );
					return false;
				}
			}
		}
		
		copy.clear();
		copy = tree;

		{
			Map::iterator mi = map.begin();
			Tree::iterator ti = copy.begin();

			for( ; mi != map.end() && ti != copy.end(); ++mi, ++ti )
			{
				if( mi->first != ti->first )
				{
					status << "Assign failed, map key " << mi->first
						<< " does not match tree key " << ti->first << "\n";
					dumpTree( copy, path );
					return false;
				}
				if( mi->second != ti->second )
				{
					status << "Assign failed, map value " 
						<< mi->second
						<< " does not match tree value " << ti->second << "\n";
					dumpTree( copy, path );
					return false;
				}
			}
		}
	
		status << "  Test Copy Passed.\n";
		return true;
	}
	
	void TestBTree::doBenchmark()
	{
		Tree tree;
		Vector vector( elements );
		_init( vector );
				
		for( Vector::iterator fi = vector.begin(); fi != vector.end(); ++fi )
		{
			std::pair< Tree::iterator, bool > insertion 
				= tree.insert( std::make_pair( *fi, fi - vector.begin() ) );
			if( insertion.second )
			{
				std::stringstream stream;
				stream << (fi - vector.begin());
				dumpTree( tree, path, stream.str() );
			}
		}
		
		for( Vector::iterator fi = vector.begin(); fi != vector.end(); ++fi )
		{
			Tree::iterator ti = tree.find( *fi );
			if( ti != tree.end() )
			{
				tree.erase( ti );
				std::stringstream stream;
				stream << (fi - vector.begin() + elements);
				dumpTree( tree, path, stream.str() );
			}
		}	
	}

	bool TestBTree::doTest()
	{
		if( benchmark )
		{
			doBenchmark();
			return true;
		}
		else
		{
			return testRandom() && testClear() && testIteration() 
				&& testComparisons() && testSearching() && testSwap() 
				&& testInsert() && testErase() && testCopy();
		}
	}

	TestBTree::TestBTree()
	{
		name = "TestBTree";
		description = "A unit test for the Tree mapping data structure. ";
		description += "Test Points: 1) Randomly insert and remove elements ";
		description += "from a std::map and a Map assert that the final ";
		description += "versions have the exact same elements stored in the ";
		description += "same order. 2) Add elements and then clear the Map. ";
		description += " Assert that there are no elements after the clear and";
		description += " that the correct number is reported by size after ";
		description += "each insertion. 3) Test iteraion through the Map.";
		description += "4) Test each of the comparison ";
		description += "operators. 5) Test searching functions. 6) Test ";
		description += "swapping with another map 7) Test all of the insert ";
		description += "functions. 8) Test all of the erase functions. 9) ";
		description += "Test assignment and copy constructors. 10) Do not ";
		description += "run any tests, simply add a ";
		description += "sequence to the Map and write it out to graph viz ";
		description += "files after each operaton.";
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestBTree test;
	parser.description( test.testDescription() );

	parser.parse( "-s", "--seed", test.seed, 0, 
		"Seed for random tests, 0 implies seed with time." );
	parser.parse( "-v", "--verbose", test.verbose, false, 
		"Print out info after the test." );
	parser.parse( "-b", "--benchmark", test.benchmark, false,
		"Rather than testing, print out the tree after several changes." );
	parser.parse( "-e", "--elements", test.elements, 30,
		"The number of elements to add to the tree." );
	parser.parse( "-i", "--iterations", test.iterations, 1000,
		"The number of iterations to perform tests." );
	parser.parse( "-p", "--path", test.path, "temp",
		"Relative path to store graphviz representations of the tree." );
	parser.parse();
	
	test.test();
	
	return test.passed();
}

#endif

