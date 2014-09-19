/*!
	\file TextureOperations.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday September 29, 2009
	\brief Implementations of texture operations.
*/

#ifndef TEXTURE_OPERATIONS_CPP_INCLUDED
#define TEXTURE_OPERATIONS_CPP_INCLUDED

#include <ocelot/executive/interface/TextureOperations.h>

namespace executive
{
	namespace tex
	{
		ir::PTXF64 wrap( ir::PTXF64 b, unsigned int limit, 
			ir::Texture::AddressMode mode ) {
			if (mode == ir::Texture::Wrap) {
				if (b < 0) {
					b = fmod(-b, limit);
					b = limit - b;
				}
				else {
					b = fmod(b, limit);
				}
				if (b < 0) {
					b = limit - 1 + b;
				} else if (b > limit - 1) {
					b = b - limit + 1;
				}
			}
			else {
				b = std::max(b, 0.0);
				b = std::min(b, (double)(limit - 1));
			}
	
			return b;
		}
		
		ir::PTXF32 channelReadF32( const ir::Texture& texture, 
			unsigned int shift, unsigned int mask, unsigned int index )
		{
			unsigned int bits = texture.x + texture.y + texture.z + texture.w;
			unsigned int bytes = bits / 8;
			unsigned int offset = shift / 8;
			
			float value = *((ir::PTXF32*)(((ir::PTXB8*) texture.data) 
				+ index*bytes + offset));
			mask &= hydrazine::bit_cast< unsigned int >( value );
			return hydrazine::bit_cast< ir::PTXF32 >( mask );
		}
				
		static ir::PTXU64 channelReadAddress( const ir::Texture& texture,
			unsigned int index )
		{
			unsigned int bits = texture.x + texture.y + texture.z + texture.w;
			unsigned int bytes = bits / 8;
			return ((ir::PTXU64) texture.data) + index*bytes;
		}
		
		void addresses( const ir::Texture& texture, ir::PTXF64 b, 
			trace::TraceEvent::U64Vector& addresses )
		{
			if (texture.normalize) 
			{
				b = b * texture.size.x;
			}
		
			if (texture.interpolation == ir::Texture::Nearest) 
			{
				ir::PTXF64 index = (ir::PTXF64)b;
				unsigned int windex = wrap(index, texture.size.x, 
					texture.addressMode[0]);
				addresses.push_back(channelReadAddress(texture, 
					windex));
			} 
			else {
				b -= 0.5f;

				ir::PTXF64 low = floor(b);
				ir::PTXF64 high = floor(b + 1);
				unsigned int wlow = wrap(low, texture.size.x, 
					texture.addressMode[0]);
				unsigned int whigh = wrap(high, texture.size.x, 
					texture.addressMode[0]);

				addresses.push_back(channelReadAddress(texture,
					wlow));
				addresses.push_back(channelReadAddress(texture,
					whigh));
			}
		}
		
		void addresses( const ir::Texture& texture, ir::PTXF64 b0,
			ir::PTXF64 b1, trace::TraceEvent::U64Vector& addresses )
		{
			ir::PTXF64 b[2] = { b0, b1 };
			
			if (texture.normalize) {
				b[0] = b[0] * texture.size.x;
				b[1] = b[1] * texture.size.y;
			}		
			
			if (texture.interpolation == ir::Texture::Nearest) {
				ir::PTXF64 index[2] = { ( ir::PTXF64 )(ir::PTXS64)b[0], 
					( ir::PTXF64 )(ir::PTXS64)b[1] };
				unsigned int windex[2];
				windex[0] = wrap(index[0], texture.size.x, 
					texture.addressMode[0]);
				windex[1] = wrap(index[1], texture.size.y, 
					texture.addressMode[1]);
				
				addresses.push_back(channelReadAddress(texture, 
					windex[0] + windex[1] * texture.size.x));
			} 
			else {
				b[0] -= 0.5f;
				b[1] -= 0.5f;
			
				ir::PTXF64 low[2] = {floor(b[0]), floor(b[1])};
				ir::PTXF64 high[2] = {floor(b[0] + 1), floor(b[1] + 1)};
				unsigned int wlow[2];
				unsigned int whigh[2];
				wlow[0] = wrap(low[0], texture.size.x, texture.addressMode[0]);
				wlow[1] = wrap(low[1], texture.size.y, texture.addressMode[1]);
				whigh[0] = wrap(high[0], texture.size.x,
					texture.addressMode[0]);
				whigh[1] = wrap(high[1], texture.size.y,
					texture.addressMode[1]);
				
				addresses.push_back(channelReadAddress(texture, 
					wlow[0] + texture.size.x * wlow[1]));
				addresses.push_back(channelReadAddress(texture, 
					whigh[0] + texture.size.x * whigh[1]));
				addresses.push_back(channelReadAddress(texture, 
					wlow[0] + texture.size.x * whigh[1]));
				addresses.push_back(channelReadAddress(texture, 
					whigh[0] + texture.size.x * wlow[1]));
			}
		}
		
		void addresses( const ir::Texture& texture, ir::PTXF64 b0, 
			ir::PTXF64 b1, ir::PTXF64 b2,
			trace::TraceEvent::U64Vector& addresses )
		{
			ir::PTXF64 b[3] = {b0, b1, b2};
			
			if (texture.normalize) {
				b[0] = b[0] * texture.size.x;
				b[1] = b[1] * texture.size.y;
				b[2] = b[2] * texture.size.z;
			}

			if (texture.interpolation == ir::Texture::Nearest) {
				ir::PTXF64 index[3] = { (ir::PTXF64)b[0], (ir::PTXF64)b[1], 
					(ir::PTXF64)b[2]};
				unsigned int windex[3];
				windex[0] = wrap(index[0], texture.size.x,
					texture.addressMode[0]);
				windex[1] = wrap(index[1], texture.size.y,
					texture.addressMode[1]);
				windex[2] = wrap(index[2], texture.size.z,
					texture.addressMode[2]);
				
				addresses.push_back(channelReadAddress(texture, 
					windex[0] + windex[1]*texture.size.x 
					+ index[2]*texture.size.x*texture.size.y));
			}
			else {
				b[0] -= 0.5f;
				b[1] -= 0.5f;
				b[2] -= 0.5f;

				ir::PTXF64 low[3] = {floor(b[0]), floor(b[1]), floor(b[2])};
				ir::PTXF64 high[3] = {floor(b[0] + 1), floor(b[1] + 1), 
					floor(b[2] + 1)};
				unsigned int wlow[3];
				unsigned int whigh[3];
				wlow[0] = wrap(low[0], texture.size.x, texture.addressMode[0]);
				wlow[1] = wrap(low[1], texture.size.y, texture.addressMode[1]);
				wlow[2] = wrap(low[2], texture.size.z, texture.addressMode[2]);
				whigh[0] = wrap(high[0], texture.size.x, 
					texture.addressMode[0]);
				whigh[1] = wrap(high[1], texture.size.y, 
					texture.addressMode[1]);
				whigh[2] = wrap(high[2], texture.size.z, 
					texture.addressMode[2]);

				addresses.push_back(channelReadAddress(texture, 
					wlow[0] + texture.size.x * wlow[1] 
					+ texture.size.x * texture.size.y * wlow[2]));
				addresses.push_back(channelReadAddress(texture, 
					wlow[0] + texture.size.x * wlow[1] 
					+ texture.size.x * texture.size.y * whigh[2]));
				addresses.push_back(channelReadAddress(texture, 
					wlow[0] + texture.size.x * whigh[1] 
					+ texture.size.x * texture.size.y * wlow[2]));
				addresses.push_back(channelReadAddress(texture, 
					wlow[0] + texture.size.x * whigh[1] 
					+ texture.size.x * texture.size.y * whigh[2]));
				addresses.push_back(channelReadAddress(texture, 
					whigh[0] + texture.size.x * wlow[1] 
					+ texture.size.x * texture.size.y * wlow[2]));
				addresses.push_back(channelReadAddress(texture, 
					whigh[0] + texture.size.x * wlow[1] 
					+ texture.size.x * texture.size.y * whigh[2]));
				addresses.push_back(channelReadAddress(texture, 
					whigh[0] + texture.size.x * whigh[1] 
					+ texture.size.x * texture.size.y * wlow[2]));
				addresses.push_back(channelReadAddress(texture, 
					whigh[0] + texture.size.x * whigh[1] 
					+ texture.size.x * texture.size.y * whigh[2]));
			}
		}
				
	}
}

#endif

