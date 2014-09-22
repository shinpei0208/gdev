/*!
	\file TextureOperations.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday September 29, 2009
	\brief Implementations of texture operations.
*/

#ifndef TEXTURE_OPERATIONS_H_INCLUDED
#define TEXTURE_OPERATIONS_H_INCLUDED

#include <ocelot/ir/interface/Texture.h>
#include <ocelot/trace/interface/TraceEvent.h>

#include <cmath>
#include <cfloat>

#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/debug.h>

#include <iostream>

namespace executive
{
	/*! \brief A namespace for texture sampling instructions */
	namespace tex
	{
		ir::PTXF64 wrap( ir::PTXF64 b, unsigned int limit, 
			ir::Texture::AddressMode mode );

		ir::PTXF32 channelReadF32( const ir::Texture& texture, 
			unsigned int shift, unsigned int mask, unsigned int index );

		template<typename D> D channelRead( const ir::Texture& texture, 
			unsigned int shift, unsigned int mask, unsigned int index )
		{
			unsigned int bits = texture.x + texture.y + texture.z + texture.w;
			unsigned int bytes = bits / 8;
			unsigned int offset = shift / 8;
			D value = *((D*)(((ir::PTXB8*) texture.data) 
				+ index*bytes + offset));
			value &= mask;
			return value;
		}

		template<unsigned int dim>
		void getShiftAndMask( unsigned int &shift, ir::PTXB64& mask,
			const ir::Texture& texture )
		{
			mask = 1;

			switch (dim) 
			{
				case 0: mask <<= (texture.x);
					--mask; 
					shift = 0; 
					break;
				case 1: mask <<= (texture.y);
					--mask;
					shift = texture.x; 
					break;
				case 2: mask <<= (texture.z);
					--mask; 
					shift = texture.x + texture.y; 
					break;
				case 3: mask <<= (texture.w);
					--mask; 
					shift = texture.z + texture.y + texture.x; 
					break;
				default: assert("Invalid texture index" == 0); 
					break;
			}
		}

		/*! \brief sample in one dimension */
		template<unsigned int dim, typename D, typename B>
		D sample( const ir::Texture& texture, B b0 ) 
		{
			D d = 0;
			ir::PTXF64 b = ( ir::PTXF64 ) b0;
			ir::PTXB64 mask;
			unsigned int shift;

			getShiftAndMask<dim>(shift, mask, texture);

			if (texture.normalize) 
			{
				b = b * texture.size.x;
			}
		
			if (texture.interpolation == ir::Texture::Nearest) 
			{
				ir::PTXF64 index = (ir::PTXF64)b;
				unsigned int windex = wrap(index, texture.size.x, 
					texture.addressMode[0]);
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXU32 result = channelRead<ir::PTXU32>(texture, 
							shift, mask, windex);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXS32 result = channelRead<ir::PTXS32>(texture, 
							shift, mask, windex);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, 
							shift, mask, windex);
						d = result;
						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}
			} 
			else {
				b -= 0.5f;

				ir::PTXF64 low = floor(b);
				ir::PTXF64 high = floor(b + 1);
				unsigned int wlow = wrap(low, texture.size.x, 
					texture.addressMode[0]);
				unsigned int whigh = wrap(high, texture.size.x, 
					texture.addressMode[0]);
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXU64 result = channelRead<ir::PTXU32>(texture,
							shift, mask, wlow) * (high - b);
						result += channelRead<ir::PTXU32>(texture, shift, 
							mask, whigh) * (b - low);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXS64 result = channelRead<ir::PTXS32>(texture,
							shift, mask, wlow) * (high - b);
						result += channelRead<ir::PTXS32>(texture, shift, mask, 
							whigh) * (b - low);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, shift, 
							mask, wlow) * (high - b);
						result += channelReadF32(texture, shift, 
							mask, whigh) * (b - low);
						d = result;
						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}
			}

			if(texture.normalizedFloat)
			{
				ir::PTXF32 f = ( d + 0.0 ) / (mask + 1);
				d = hydrazine::bit_cast< D >( f );
			}
	
			return d;
		}

		/*!	\brief sample in 2 dimensions */
		template<unsigned int dim, typename D, typename B>
		D sample(const ir::Texture& texture, B b0, B b1) {
			D d = 0;
			ir::PTXF64 b[2] = { ( ir::PTXF64 ) b0, ( ir::PTXF64 ) b1 };
			ir::PTXB64 mask;
			unsigned int shift;

			getShiftAndMask<dim>(shift, mask, texture);
	
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
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXU32 result = channelRead<ir::PTXU32>(texture, 
							shift, mask, 
							windex[0] + windex[1] * texture.size.x);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXS32 result = channelRead<ir::PTXS32>(texture, 
							shift, mask, 
							windex[0] + windex[1] * texture.size.x);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, shift, mask, 
							windex[0] + windex[1] * texture.size.x);
						d = result;

						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}

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
				whigh[0] = wrap(high[0], texture.size.x, texture.addressMode[0]);
				whigh[1] = wrap(high[1], texture.size.y, texture.addressMode[1]);
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXF64 result = channelRead<ir::PTXU32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1]) * (high[0] - b[0]) 
							* (high[1] - b[1]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1]) * (b[0] - low[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1]) * (high[0] - b[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * wlow[1]) * (b[0] - low[0]) 
							* (high[1] - b[1]);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXF64 result = channelRead<ir::PTXS32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1]) * (high[0] - b[0]) 
							* (high[1] - b[1]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1]) * (b[0] - low[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1]) * (high[0] - b[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, high[0] 
							+ texture.size.x * low[1]) * (b[0] - low[0]) 
							* (high[1] - b[1]);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1]) * (high[0] - b[0]) 
							* (high[1] - b[1]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * whigh[1]) * (b[0] - low[0]) 
							* (b[1] - low[1]);
						result += channelReadF32(texture, shift, mask, wlow[0] 
							+ texture.size.x * whigh[1]) * (high[0] - b[0]) 
							* (b[1] - low[1]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * wlow[1]) * (b[0] - low[0]) 
							* (high[1] - b[1]);
						d = result;
						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}
			}

			if(texture.normalizedFloat)
			{
				ir::PTXF32 f = ( d + 0.0 ) / (mask + 1);
				d = hydrazine::bit_cast< D >( f );
			}
	
			return d;
		}


		template<unsigned int dim, typename D, typename B>
		D sample(const ir::Texture& texture, B b0, B b1, B b2) {
			D d = 0;
			ir::PTXF64 b[3] = {( ir::PTXF64 ) b0, ( ir::PTXF64 ) b1, 
				( ir::PTXF64 ) b2};
			ir::PTXB64 mask;
			unsigned int shift;

			getShiftAndMask<dim>(shift, mask, texture);
	
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
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXU32 result = channelRead<ir::PTXU32>(texture, 
							shift, mask, 
							windex[0] + windex[1]*texture.size.x 
							+ index[2]*texture.size.x*texture.size.y);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXS32 result = channelRead<ir::PTXS32>(texture, 
							shift, mask, 
							windex[0] + windex[1]*texture.size.x 
							+ windex[2]*texture.size.x*texture.size.y);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, 
							shift, mask, 
							windex[0] + windex[1]*texture.size.x 
							+ windex[2]*texture.size.x*texture.size.y);

						d = result;
						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}
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
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXF64 result = channelRead<ir::PTXU32>(texture, 
							shift, mask, 
							wlow[0] + texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (high[0] - b[0]) * (high[1] - b[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXU32>(texture, shift, 
							mask, wlow[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (high[0] - b[0]) * (high[1] - b[1]) 
							* (b[2] - low[2]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (high[0] - b[0]) * (b[1] - low[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (high[0] - b[0]) * (b[1] - low[1]) 
							* (b[2] - low[2]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (b[0] - low[0]) * (high[1] - b[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (b[0] - low[0]) * (high[1] - b[1]) 
							* (b[2] - low[2]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (b[0] - low[0]) * (b[1] - low[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (b[0] - low[0]) * (b[1] - low[1]) 
							* (b[2] - low[2]);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXF64 result = channelRead<ir::PTXS32>(texture, 
							shift, mask, 
							wlow[0] + texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (high[0] - b[0]) * (high[1] - b[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (high[0] - b[0]) * (high[1] - b[1]) 
							* (b[2] - low[2]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (high[0] - b[0]) * (b[1] - low[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (high[0] - b[0]) * (b[1] - low[1]) 
							* (b[2] - low[2]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (b[0] - low[0]) * (high[1] - b[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (b[0] - low[0]) * (high[1] - b[1]) 
							* (b[2] - low[2]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (b[0] - low[0]) * (b[1] - low[1]) 
							* (high[2] - b[2]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (b[0] - low[0]) * (b[1] - low[1]) 
							* (b[2] - low[2]);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (high[0] - b[0]) * (high[1] - b[1]) 
							* (high[2] - b[2]);
						result += channelReadF32(texture, shift, mask, wlow[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (high[0] - b[0]) * (high[1] - b[1]) 
							* (b[2] - low[2]);
						result += channelReadF32(texture, shift, mask, wlow[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (high[0] - b[0]) * (b[1] - low[1]) 
							* (high[2] - b[2]);
						result += channelReadF32(texture, shift, mask, wlow[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (high[0] - b[0]) * (b[1] - low[1]) 
							* (b[2] - low[2]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (b[0] - low[0]) * (high[1] - b[1]) 
							* (high[2] - b[2]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * wlow[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (b[0] - low[0]) * (high[1] - b[1]) 
							* (b[2] - low[2]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * wlow[2]) 
							* (b[0] - low[0]) * (b[1] - low[1]) 
							* (high[2] - b[2]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * whigh[1] 
							+ texture.size.x * texture.size.y * whigh[2]) 
							* (b[0] - low[0]) * (b[1] - low[1]) 
							* (b[2] - low[2]);

						d = result;
						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}
			}
	
			if( texture.normalizedFloat )
			{
				ir::PTXF32 f = ( d + 0.0 ) / (mask + 1);
				d = hydrazine::bit_cast< D >( f );
			}
	
			return d;
		}
		
		template <typename T> T fabs(T t) {
			if (t < 0) return -t;
			return t;
		}
		template <typename T> T signum(T t) {
			if (t < 0) return -1;
			if (t > 0) return 1;
			return 0;
		}
		
		template<unsigned int dim, typename D>
		D sampleCube(const ir::Texture& texture, float b0, float b1, float b2) {
			D d = 0;
			ir::PTXF32 b[3] = {b0,b1,b2};
			ir::PTXB64 mask;
			unsigned int shift;

			getShiftAndMask<dim>(shift, mask, texture);
				
			// find which is largest in magnitude
			float magnitudes[3] = {fabs(b[0]), fabs(b[1]), fabs(b[2])};
			float signs[3] = {signum(b[0]), signum(b[1]), signum(b[2])};
			int maxDim = 0;
			for (int i = 1; i < 3; i++) {
				if (magnitudes[i] > magnitudes[maxDim]) {
					maxDim = i;
				}
			}
			int textureIndex = maxDim*2 + (signs[maxDim] < 0 ? 1 : 0);
			float layerCoords[3] = {0, 0};
			layerCoords[0] = magnitudes[maxDim];
			switch (textureIndex) {
			case 0:
				layerCoords[1] = -b[2];
				layerCoords[2] = -b[1];
				break;
			case 1:
				layerCoords[1] = b[2];
				layerCoords[2] = -b[1];
			break;
			case 2:
				layerCoords[1] = b[0];
				layerCoords[2] = b[2];
			break;
			case 3:
				layerCoords[1] = b[0];
				layerCoords[2] = -b[2];
			break;
			case 4:
				layerCoords[1] = b[0];
				layerCoords[2] = -b[1];
			break;
			case 5:
				layerCoords[1] = -b[0];
				layerCoords[2] = -b[1];
			break;
			default:
			break;
			}
			b[0] = (layerCoords[1] / layerCoords[0] + 1.0f) / 2.0f;
			b[1] = (layerCoords[2] / layerCoords[0] + 1.0f) / 2.0f;
			
			if (texture.normalize) {
				b[0] = b[0] * texture.size.x;
				b[1] = b[1] * texture.size.y;
			}
			
			if (texture.interpolation == ir::Texture::Nearest) {
				ir::PTXF64 index[2] = { (ir::PTXF64)b[0], (ir::PTXF64)b[1]};
				unsigned int windex[2];
				windex[0] = wrap(index[0], texture.size.x, texture.addressMode[0]);
				windex[1] = wrap(index[1], texture.size.y, texture.addressMode[1]);
				
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXU32 result = channelRead<ir::PTXU32>(texture, 
							shift, mask, 
							windex[0] + windex[1]*texture.size.x + texture.size.x*texture.size.y*textureIndex);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXS32 result = channelRead<ir::PTXS32>(texture, 
							shift, mask, 
							windex[0] + windex[1]*texture.size.x + texture.size.x*texture.size.y*textureIndex);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, 
							shift, mask, 
							windex[0] + windex[1]*texture.size.x + texture.size.x*texture.size.y*textureIndex);

						d = result;
						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}
			}
			else if (texture.interpolation == ir::Texture::Linear) {
				b[0] -= 0.5f;
				b[1] -= 0.5f;
				
				ir::PTXF64 low[2] = {floor(b[0]), floor(b[1])};
				ir::PTXF64 high[2] = {floor(b[0] + 1), floor(b[1] + 1)};
				unsigned int wlow[3];
				unsigned int whigh[3];
				wlow[0] = wrap(low[0], texture.size.x, texture.addressMode[0]);
				wlow[1] = wrap(low[1], texture.size.y, texture.addressMode[1]);
				whigh[0] = wrap(high[0], texture.size.x, texture.addressMode[0]);
				whigh[1] = wrap(high[1], texture.size.y, texture.addressMode[1]);
				switch (texture.type) {
					case ir::Texture::Unsigned:
					{
						ir::PTXF64 result = channelRead<ir::PTXU32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1]) * (high[0] - b[0]) 
							* (high[1] - b[1]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1]) * (b[0] - low[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1]) * (high[0] - b[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXU32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * wlow[1]) * (b[0] - low[0]) 
							* (high[1] - b[1]);
						d = result;
						break;
					}
					case ir::Texture::Signed:
					{
						ir::PTXF64 result = channelRead<ir::PTXS32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1]) * (high[0] - b[0]) 
							* (high[1] - b[1]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, whigh[0] 
							+ texture.size.x * whigh[1]) * (b[0] - low[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * whigh[1]) * (high[0] - b[0]) 
							* (b[1] - low[1]);
						result += channelRead<ir::PTXS32>(texture, 
							shift, mask, high[0] 
							+ texture.size.x * low[1]) * (b[0] - low[0]) 
							* (high[1] - b[1]);
						d = result;
						break;
					}
					case ir::Texture::Float:
					{
						ir::PTXF32 result = channelReadF32(texture, 
							shift, mask, wlow[0] 
							+ texture.size.x * wlow[1]) * (high[0] - b[0]) 
							* (high[1] - b[1]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * whigh[1]) * (b[0] - low[0]) 
							* (b[1] - low[1]);
						result += channelReadF32(texture, shift, mask, wlow[0] 
							+ texture.size.x * whigh[1]) * (high[0] - b[0]) 
							* (b[1] - low[1]);
						result += channelReadF32(texture, shift, mask, whigh[0] 
							+ texture.size.x * wlow[1]) * (b[0] - low[0]) 
							* (high[1] - b[1]);
						d = result;
						break;
					}
					default:
						assert("Invalid texture data type" == 0);
				}
			}
			else {
				assert(0 && "invalid texture interpolation mode");
			}
			
			if( texture.normalizedFloat )
			{
				ir::PTXF32 f = ( d + 0.0 ) / (mask + 1);
				d = hydrazine::bit_cast< D >( f );
			}
	
			return d;
		}
		
		void addresses( const ir::Texture& texture, ir::PTXF64 b0,
			trace::TraceEvent::U64Vector& );
		void addresses( const ir::Texture& texture, ir::PTXF64 b0,
			ir::PTXF64 b1, trace::TraceEvent::U64Vector& );
		void addresses( const ir::Texture& texture, ir::PTXF64 b0,
			ir::PTXF64 b1, ir::PTXF64 b2, trace::TraceEvent::U64Vector& );
		
	}
}

#endif

