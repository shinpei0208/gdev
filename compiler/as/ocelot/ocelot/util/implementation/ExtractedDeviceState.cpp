/*!
	\file ExtractedDeviceState.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date 31 Jan 2011
	\brief Data structure describing device state with serialization and deserialization procedures
*/

// C++ includes
#include <iomanip>
#include <sstream>
#include <cassert>
#include <iostream>
#include <cstring>

// Boost includes
#include  <boost/lexical_cast.hpp>

// Hydrazine includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Casts.h>

// Ocelot includes
#include <ocelot/util/interface/ExtractedDeviceState.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

////////////////////////////////////////////////////////////////////////////////

// whether debugging messages are printed
#define REPORT_BASE 0

////////////////////////////////////////////////////////////////////////////////

template <typename ElemT>
struct HexTo {
	ElemT value;
	operator ElemT() const {return value;}
	friend std::istream& operator>>(std::istream& in, HexTo& out) {
		  in >> std::hex >> out.value;
		  return in;
	}
};

////////////////////////////////////////////////////////////////////////////////

static void emitEscapedString(std::ostream &out, const std::string &str) {
	for (std::string::const_iterator c_it = str.begin();
		c_it != str.end(); ++c_it) {
		if (*c_it == '"') {
			out << "\\\"";
		}
		else {
			out << *c_it;
		}
	}
}

static std::ostream & serialize(std::ostream &out, const ir::Dim3 &dim) {
	out << "[" << dim.x << ", " << dim.y << ", " << dim.z << "]";
	return out;
}

static void deserialize(ir::Dim3 &dim, const hydrazine::json::Visitor& array) {
	dim.x = array[0];
	dim.y = array[1];
	dim.z = array[2];
}

static std::ostream & serialize(std::ostream &out,
	const std::vector<int> &ints) {
	out << "[";
	int n=0;
	for (std::vector<int>::const_iterator i_it = ints.begin();
		i_it != ints.end(); ++i_it) {
		out << (n++ ? "," : "") << *i_it;
	}
	out << "]";
	return out;
}


static void deserialize(std::vector<int>& ints,
	const hydrazine::json::Visitor& array) {
	
	const hydrazine::json::DenseArray& denseArray =
		static_cast<const hydrazine::json::DenseArray&>(*array.value);
	
	size_t size = denseArray.sequence.size();
	
	ints.clear();
	ints.resize(size);
	
	std::memcpy(ints.data(), denseArray.sequence.data(), size);
}

static void serializeBinary(std::ostream &out, const size_t size,
	const char *bytes, bool raw) {
	const size_t wordSize = 4;
	if (!raw) {
		out << std::setbase(16);
		out << std::setw(2*wordSize)
			<< std::setfill('0');
	}
	for (size_t n = 0; n < size; n += wordSize) {
		unsigned int word = 0;
		for (size_t j = 0; j < wordSize; j++) {
			if (j+n < size) {
				word |= (((unsigned int)bytes[j+n] & 0x0ff) << (j * 8));
			}
		}

		if (n) {
			out << ", ";
		}

		if (!((n) % (8 * wordSize))) {
			out << "\n";
		}
		
		out << (raw ? "" : "\"0x");
		
		out << word << (raw ? "" : "\"");
	}	
	out << std::setbase(10);
}

static void serializeBinary(std::ostream &out, const size_t size,
	const char *bytes) {
	out << "{ \"bytes\": " << std::setbase(10) << size << ", \"image\": [\n";
	serializeBinary(out, size, bytes, true);
	out << std::setbase(10) << "\n] }";
}

static void deserializeBinary(util::ByteVector &bytes,
	const hydrazine::json::Array *arrayPtr, size_t size) {
	size_t wordSize = 4;
	bytes.clear();
	bytes.reserve(size);
	for (hydrazine::json::Array::ValueVector::const_iterator
		it = arrayPtr->begin(); it != arrayPtr->end(); ++it) {
		std::string wordString = (*it)->as_string();		
		unsigned int word =
			boost::lexical_cast<HexTo<unsigned int> >(wordString);
		for (size_t i = 0; i < wordSize; i++) {
			bytes.push_back(word & 0x0ff);
			word >>= 8;
		}
	}
	bytes.resize(size, 0);
}

static void deserializeBinary(util::ByteVector &bytes,
	const hydrazine::json::DenseArray *arrayPtr, size_t size) {
	bytes.resize(size);
	
	std::memcpy(bytes.data(), arrayPtr->sequence.data(), size);
}

static void deserializeBinary(util::ByteVector &bytes,
	const hydrazine::json::Visitor &object) {
	size_t size = object.parse<int>("bytes", 0);
	if (hydrazine::json::Value *arrayValue = object.find("image")) {
		if (arrayValue->type == hydrazine::json::Value::DenseArray) {
			deserializeBinary(bytes,
				static_cast<hydrazine::json::DenseArray *>(arrayValue), size);
		}
		else {
			deserializeBinary(bytes,
				static_cast<hydrazine::json::Array *>(arrayValue), size);
		}
	}
}

static void serializeBinary(std::ostream &out, const util::ByteVector &data) {
	serializeBinary(out, data.size(), &data[0]);
}


////////////////////////////////////////////////////////////////////////////////

util::ExtractedDeviceState::MemoryAllocation::MemoryAllocation(const void *ptr,
	size_t _size): 
	devicePointer(ptr), data(_size) {
}

size_t util::ExtractedDeviceState::MemoryAllocation::size() const {
	return data.size();
}

void util::ExtractedDeviceState::MemoryAllocation::serialize(
	std::ostream &out) const {
	out << "{";
	out << "  \"device\": \"" << devicePointer << "\",\n";
	out << "  \"data\": ";
	::serializeBinary(out, data);
	
	out << "}";
}

void util::ExtractedDeviceState::MemoryAllocation::deserialize(
	const hydrazine::json::Visitor &object) {
	devicePointer = hydrazine::bit_cast<const void *>(
		boost::lexical_cast<HexTo<size_t> >(
		object.parse<std::string>("device", "0x00")));
	
	if (hydrazine::json::Value *dataMemory = object.find("data")) {
		deserializeBinary(data, hydrazine::json::Visitor(dataMemory));
	}
}

////////////////////////////////////////////////////////////////////////////////

util::ExtractedDeviceState::GlobalAllocation::GlobalAllocation(const void *ptr,
	size_t _size, const std::string& m, const std::string& g): 
	module(m), name(g), data(_size) {
}

size_t util::ExtractedDeviceState::GlobalAllocation::size() const {
	return data.size();
}

void util::ExtractedDeviceState::GlobalAllocation::serialize(
	std::ostream &out) const {
	out << "{";
	out << "  \"module\": \"" << module << "\",\n";
	out << "  \"name\": \""   << name   << "\",\n";
	out << "  \"data\": ";
	::serializeBinary(out, data);
	
	out << "}";
}

void util::ExtractedDeviceState::GlobalAllocation::deserialize(
	const hydrazine::json::Visitor &object) {
	
	module = object.parse<std::string>("module", "unknown-module");
	name   = object.parse<std::string>("name",   "unknown-variable");
	
	if (hydrazine::json::Value *dataMemory = object.find("data")) {
		deserializeBinary(data, hydrazine::json::Visitor(dataMemory));
	}
}

			
std::string util::ExtractedDeviceState::GlobalAllocation::key() const {
	return module + ":" + name;
}

////////////////////////////////////////////////////////////////////////////////

void util::ExtractedDeviceState::KernelLaunch::serialize(
	std::ostream &out) const {
	out << "{ \"module\": \"" << moduleName << "\", \"kernel\": \""
		<< kernelName << "\",\n";
	out << "  \"gridDim\": "; ::serialize(out, gridDim); out << ",\n";
	out << "  \"blockDim\": "; ::serialize(out, blockDim); out << ",\n";
	out << "  \"sharedMemorySize\": " << sharedMemorySize << ",\n";
	out << "  \"staticSharedMemorySize\": " << staticSharedMemorySize << ",\n";
	out << "  \"parameterMemory\": ";
	serializeBinary(out, parameterMemory);
	out << "}";
}

void util::ExtractedDeviceState::KernelLaunch::deserialize(
	const hydrazine::json::Visitor &object) {
	moduleName = object.parse<std::string>("module", "unknown-module");
	kernelName = object.parse<std::string>("kernel", "unknown-kernel");
	
	::deserialize(gridDim, object["gridDim"]);
	::deserialize(blockDim, object["blockDim"]);
	sharedMemorySize = object.parse<int>("sharedMemorySize", 0);
	staticSharedMemorySize = object.parse<int>("staticSharedMemorySize", 0);
	
	if (hydrazine::json::Value *parameterMemory =
		object.find("parameterMemory")) {
		deserializeBinary(this->parameterMemory,
			hydrazine::json::Visitor(parameterMemory));
	}
}

////////////////////////////////////////////////////////////////////////////////

util::ExtractedDeviceState::Module::Module() {

}

util::ExtractedDeviceState::Module::~Module() {
	clear();
}

void util::ExtractedDeviceState::Module::clear() {
	for(TextureMap::iterator texture = textures.begin();
		texture != textures.end(); ++texture) {
		delete texture->second;
	}
	
	textures.clear();
}

void util::ExtractedDeviceState::Module::serializeTexture(
	const ir::Texture &texture, 
	std::ostream &out, 
	const std::string & prefix) const {

	std::vector<int> bits;
	bits.push_back(texture.x);
	bits.push_back(texture.y);
	bits.push_back(texture.z);
	bits.push_back(texture.w);

	out << "{\n";
	out << "  \"name\": \"" << texture.name << "\",\n";
	out << "  \"bits\": "; ::serialize(out, bits); out << ",\n";
	out << "  \"normalize\": "
		<< (texture.normalize ? "true" : "false") << ",\n";
	out << "  \"normalizedFloat\": "
		<< (texture.normalizedFloat ? "true" : "false") << ",\n";
	out << "  \"size\": "; ::serialize(out, texture.size); out << ",\n";
	out << "  \"type\": \"" << ir::Texture::toString(texture.type) << "\",\n";
	out << "  \"addressMode\": [ ";
	for (int i = 0; i < 3; i++) {
		out << (i ? ", " : "") << ir::Texture::toString(texture.addressMode[i]);
	}
	out << " ],\n";
	out << "  \"interpolation\": \""
		<< ir::Texture::toString(texture.interpolation) << "\",\n";
	out << "  \"data\": \"" << texture.data << "\"\n";
	out << "}\n";
}

void util::ExtractedDeviceState::Module::serialize(std::ostream &out,
	const std::string & prefix) const {
	out << "{\n";
	out << "  \"name\": \"" << name << "\",\n";
	out << "  \"ptx\": \"";
	emitEscapedString(out, ptx);
	out << "\"";
	if (textures.size()) {
		out << ",\n  \"textures\": [\n";
		int n = 0;
		for (TextureMap::const_iterator t_it = textures.begin();
			t_it != textures.end(); ++t_it) {
			
			if (n++) { out << ",\n"; }
			serializeTexture(*(t_it->second), out, prefix);
		}
		out << "]\n";
	}
	out << "}\n";
}

void util::ExtractedDeviceState::Module::deserialize(
	const hydrazine::json::Visitor& object) {
	name = object.parse<std::string>("name", "module");
	ptx = object.parse<std::string>("ptx", "");

	hydrazine::json::Value* textureValue = object.find("textures");

	if (textureValue) {
		hydrazine::json::Visitor texturesArray(textureValue);
		for (hydrazine::json::Array::const_iterator
			tex_it = texturesArray.begin_array();
			tex_it != texturesArray.end_array(); ++tex_it) {
		
			ir::Texture* texture = new ir::Texture;

			deserializeTexture(*texture, hydrazine::json::Visitor(*tex_it));
		
			this->textures[texture->demangledName()] = texture;
		}
	}
}

void util::ExtractedDeviceState::Module::deserializeTexture(
	ir::Texture &texture, 
	const hydrazine::json::Visitor& object) {

	texture.name = object.parse<std::string>("name", "unknown-texture");

	std::vector<int> bits;
	::deserialize(bits, object["bits"]);

	texture.x = bits[0];
	texture.y = bits[1];
	texture.z = bits[2];
	texture.w = bits[3];

	texture.normalize       = object.parse<bool>("normalize", false);
	texture.normalizedFloat = object.parse<bool>("normalizedFloat", false);

	::deserialize(texture.size, object["size"]);

	texture.type = ir::Texture::typeFromString(
		object.parse<std::string>("type", "Invalid"));

	hydrazine::json::Visitor modeArray(object["addressMode"]);
	texture.addressMode[0] = ir::Texture::modeFromString(modeArray[0]);
	texture.addressMode[1] = ir::Texture::modeFromString(modeArray[1]);
	texture.addressMode[2] = ir::Texture::modeFromString(modeArray[2]);

	texture.interpolation = ir::Texture::interpolationFromString(
		object.parse<std::string>("interpolation", "Invalid"));
	
	texture.data = hydrazine::bit_cast<void *>(
		boost::lexical_cast<HexTo<size_t> >(
		object.parse<std::string>("data", "0x00")));
	
}

////////////////////////////////////////////////////////////////////////////////

util::ExtractedDeviceState::Application::Application() {
	name = "cudaApp";
}

void util::ExtractedDeviceState::Application::serialize(
	std::ostream &out) const {
	out << "{\n\"name\": \"";
	emitEscapedString(out, name);
	out << "\",\n\"cudaDevice\":\""; 
	emitEscapedString(out, cudaDevice);
	out << "\"}";
}

void util::ExtractedDeviceState::Application::deserialize(
	const hydrazine::json::Visitor &object) {
	name = object.parse<std::string>("name", "cudaApp");
	cudaDevice = object.parse<std::string>("cudaDevice", "gpu");
}

////////////////////////////////////////////////////////////////////////////////

//! \brief constructs from an istream - input must be JSON document
util::ExtractedDeviceState::ExtractedDeviceState(std::istream &in) {
	deserialize(in);
}

util::ExtractedDeviceState::ExtractedDeviceState() {

}

util::ExtractedDeviceState::~ExtractedDeviceState() {
	clear();
}

//! \brief store data in host memory to file
void util::ExtractedDeviceState::serialize(std::ostream &out) const {
	// only serialize the module containing the executed kernel

	size_t n = 0;
	
	out << "{\n";
	out << "\"application\":";
	
	application.serialize(out);
	
	out << ",\n\"allocations\": [";
	n = 0;
	for (GlobalAllocationMap::const_iterator
		alloc_it = globalAllocations.begin(); 
		alloc_it != globalAllocations.end(); ++alloc_it) {
	
		out << (n++ ? ",\n":"");
		alloc_it->second->serialize(out);
	}

	out << "],\n\"global_allocations\": [";
	n = 0;
	for (GlobalVariableMap::const_iterator
		alloc_it = globalVariables.begin(); 
		alloc_it != globalVariables.end(); ++alloc_it) {
	
		out << (n++ ? ",\n":"");
		alloc_it->second->serialize(out);
	}
	
	out << "],\n\"post_launch_allocations\": [";
	n = 0;
	for (GlobalAllocationMap::const_iterator
		alloc_it = postLaunchGlobalAllocations.begin(); 
		alloc_it != postLaunchGlobalAllocations.end(); ++alloc_it) {
	
		out << (n++ ? ",\n":"");
		alloc_it->second->serialize(out);
	}

	out << "],\n\"post_launch_global_allocations\": [";
	n = 0;
	for (GlobalVariableMap::const_iterator
		alloc_it = postLaunchGlobalVariables.begin(); 
		alloc_it != postLaunchGlobalVariables.end(); ++alloc_it) {
	
		out << (n++ ? ",\n":"");
		alloc_it->second->serialize(out);
	}
		
	out << "],\n";
	
	out <<" \"kernelLaunch\": ";
	launch.serialize(out);
	out << ",\n";
	
	out << "\"modules\": [";
	
	ModuleMap::const_iterator mod_it = modules.find(launch.moduleName);
	if (mod_it == modules.end()) {
		n = 0;
		for (ModuleMap::const_iterator mod_it = modules.begin(); 
			mod_it != modules.end(); ++mod_it) {
		
			out << (n++ ? ",":"");
			mod_it->second->serialize(out, application.name);
		}
	}
	else {
		mod_it->second->serialize(out);
	}
	out << "]\n";
	out << "}\n";
}

//! \brief load data from JSON file to host memory
void util::ExtractedDeviceState::deserialize(std::istream &in) {
	hydrazine::json::Parser parser;
	hydrazine::json::Object *stateObject = 0;

	{
		stateObject = parser.parse_object(in);
		hydrazine::json::Visitor object(stateObject);
		
		application.deserialize(object["application"]);
		launch.deserialize(object["kernelLaunch"]);
		
		hydrazine::json::Visitor allocationsArray(object["allocations"]);
		for (hydrazine::json::Array::const_iterator
			alloc_it = allocationsArray.begin_array();
			alloc_it != allocationsArray.end_array(); ++alloc_it) {
			
			MemoryAllocation* allocation = new MemoryAllocation;
			allocation->deserialize(hydrazine::json::Visitor(*alloc_it));
			globalAllocations[allocation->devicePointer] = allocation;
		}
		
		hydrazine::json::Visitor globalsArray(object["global_allocations"]);
		for (hydrazine::json::Array::const_iterator
			alloc_it = globalsArray.begin_array();
			alloc_it != globalsArray.end_array(); ++alloc_it) {
			
			GlobalAllocation* allocation = new GlobalAllocation;
			allocation->deserialize(hydrazine::json::Visitor(*alloc_it));
			globalVariables[allocation->key()] = allocation;
		}
		
		hydrazine::json::Visitor postAllocationsArray(
			object["post_launch_allocations"]);
		for (hydrazine::json::Array::const_iterator
			alloc_it = postAllocationsArray.begin_array();
			alloc_it != postAllocationsArray.end_array(); ++alloc_it) {
			
			MemoryAllocation* allocation = new MemoryAllocation;
			allocation->deserialize(hydrazine::json::Visitor(*alloc_it));
			postLaunchGlobalAllocations[allocation->devicePointer] = allocation;
		}
		
		hydrazine::json::Visitor postGlobalsArray(
			object["post_launch_global_allocations"]);
		for (hydrazine::json::Array::const_iterator
			alloc_it = postGlobalsArray.begin_array();
			alloc_it != postGlobalsArray.end_array(); ++alloc_it) {
			
			GlobalAllocation* allocation = new GlobalAllocation;
			allocation->deserialize(hydrazine::json::Visitor(*alloc_it));
			postLaunchGlobalVariables[allocation->key()] = allocation;
		}

		hydrazine::json::Visitor modulesArray(object["modules"]);
		for (hydrazine::json::Array::const_iterator
			mod_it = modulesArray.begin_array();
			mod_it != modulesArray.end_array(); ++mod_it) {
			
			Module* module = new Module;
			module->deserialize(hydrazine::json::Visitor(*mod_it));
			modules[module->name] = module;
		}
	}
	
	if (stateObject) delete stateObject;
}

//! \brief clear all data structures
void util::ExtractedDeviceState::clear() {
	for (ModuleMap::iterator mod_it = modules.begin();
		mod_it != modules.end(); ++mod_it) {
		delete mod_it->second;
	}
	modules.clear();
	
	clearData();	
}

void util::ExtractedDeviceState::clearData() {
	for (GlobalAllocationMap::iterator ga_it = globalAllocations.begin();
		ga_it != globalAllocations.end(); ++ga_it) {
		
		delete ga_it->second;
	}
	globalAllocations.clear();

	for (GlobalAllocationMap::iterator
		ga_it = postLaunchGlobalAllocations.begin();
		ga_it != postLaunchGlobalAllocations.end(); ++ga_it) {
		
		delete ga_it->second;
	}
	postLaunchGlobalAllocations.clear();
	
	for (GlobalVariableMap::iterator ga_it = globalVariables.begin();
		ga_it != globalVariables.end(); ++ga_it) {
		
		delete ga_it->second;
	}
	globalVariables.clear();
	
	for (GlobalVariableMap::iterator ga_it = postLaunchGlobalVariables.begin();
		ga_it != postLaunchGlobalVariables.end(); ++ga_it) {
		
		delete ga_it->second;
	}
	postLaunchGlobalVariables.clear();
}

