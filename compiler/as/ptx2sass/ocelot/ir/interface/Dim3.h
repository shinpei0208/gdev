/*!
	\file Dim3.h
	\author Gregory Diamos <gregory.diamos@gatech>
	\date Thursday September 17, 2009
	\brief The header file for the Dim3 class
*/

#ifndef DIM_3_H_INCLUDED
#define DIM_3_H_INCLUDED

namespace ir
{
	/*! Class representing dimensions of blocks and grids */
	class Dim3 
	{
		public:
			int x; //! x dimension
			int y; //! y dimension
			int z; //! z dimension
		public:
			Dim3(int X=1, int Y=1, int Z=1);
			
			int size() const { return x * y * z; }
			bool operator==(const Dim3 &d) { return d.x == x && d.y == y && d.z == z; }
			bool operator!=(const Dim3 &d) { return d.x != x || d.y != y || d.z != z; }
	};

}

#endif

