/*!
	\file test_json.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\date 27 October 2009

	\brief test procedure for JSON parser
*/

#include <iostream>
#include <sstream>
#include <math.h>

#include <hydrazine/interface/Test.h>
#include <hydrazine/implementation/Exception.h>
#include <hydrazine/implementation/json.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/implementation/debug.h>

using namespace hydrazine;



class TestJson : public test::Test
{
public:
		
	bool test_json_string(bool verbose) {
		bool passes = true;

		json::Parser parser;

		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "\"hello, JSON world!\"";
				json::String *test_string = parser.parse_string(ss_input_string);
				if (test_string) {
					if (test_string->value_string != "hello, JSON world!") {
						passes = false;
					}
					delete test_string;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  string test 1 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "\"LIBkerr\\n\"";
				json::String *test_string = parser.parse_string(ss_input_string);
				if (test_string) {
					if (test_string->value_string != "LIBkerr\n") {
						passes = false;
					}
					delete test_string;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  string test 2 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "\"LIBkerr\\n\"";
				json::String *test_string = parser.parse_string(ss_input_string);
				if (test_string) {
					if (test_string->value_string != "LIBkerr\n") {
						passes = false;
					}
					delete test_string;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  string test 2 - " << (passes ? "passes" : "fails") << "\n";
		}

		return passes;
	}

	bool test_json_number(bool verbose) {
		bool passes = true;
		json::Parser parser;

		// integer
		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "-123";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Integer) {
						passes = false;
					}
					else if (test_number->value_integer != -123) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  integer test 1 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "0";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Integer) {
						passes = false;
					}
					else if (test_number->value_integer != 0) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  integer test 2 - " << (passes ? "passes" : "fails") << "\n";
		}

		// real
		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "-93.23";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Real) {
						passes = false;
					}
					else if (fabs(test_number->value_real + 93.23) > 0.001) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  real test 1 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "6.022e23";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Real) {
						passes = false;
					}
					else if (fabs(test_number->value_real - 6.022e23) > 1.0e20) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  real test 2 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "3.14159265";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Real) {
						passes = false;
					}
					else if (fabs(test_number->value_real - 3.14159265) > 1.0e-8) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  real test 3 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "6.6261E-34";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Real) {
						passes = false;
					}
					else if (fabs(test_number->value_real - 6.6261e-34) > 1.0e-37) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  real test 4 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "0.125";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Real) {
						passes = false;
					}
					else if (fabs(test_number->value_real - 0.125) > 1.0e-4) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  real test 5 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "0.125e-5";
				json::Number *test_number = parser.parse_number(ss_input);
				if (test_number) {
					if (test_number->number_type != json::Number::Real) {
						passes = false;
					}
					else if (fabs(test_number->value_real - 0.125e-5) > 1.0e-9) {
						passes = false;
					}
					delete test_number;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  real test 6 - " << (passes ? "passes" : "fails") << "\n";
		}

		return passes;
	}

	bool test_json_object(bool verbose) {
		bool passes = true;
		json::Parser parser;

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "{ \"name\" : \"libkerr\" }";
				json::Object *test_object = parser.parse_object(ss_input);
				if (test_object) {
					if (test_object->dictionary["name"]->type == json::Value::String) {
						json::String *string_object = static_cast<json::String *>(test_object->dictionary["name"]);
						if (string_object->value_string != "libkerr") {
							passes = false;
							if (verbose) {
								std::cout << "  test_object->dictionary[name] = " << string_object->value_string << "\n";
							}
						}
					}
					else {
						if (verbose) {
							std::cout << "  test_object->dictionary[name]->type = " 
								<< test_object->dictionary["name"]->type << "\n";
						}
						passes = false;
					}
					delete test_object;
				}
				else {
					passes = false;
					if (verbose) {
						std::cout << "  failed to parse object for some reason\n";
					}
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  object test 1 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "{ \"age\": 1 , \"name\" : \"libkerr\" , \"version\": 0.01 }";
				json::Object *test_object = parser.parse_object(ss_input);
				if (test_object) {
					if (test_object->dictionary["name"]->as_string() != "libkerr") { passes = false; }
					else if (test_object->dictionary["age"]->as_integer() != 1) { passes = false; }
					else if (test_object->dictionary["version"]->as_real() != 0.01) { passes = false; }
					delete test_object;
				}
				else {
					passes = false;
					if (verbose) {
						std::cout << "  failed to parse object for some reason\n";
					}
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  object test 2 - " << (passes ? "passes" : "fails") << "\n";
		}


		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "{\"contents\":{\"age\":1,\"name\":\"libkerr\",\"version\":0.01}}";
				json::Object *test_object = parser.parse_object(ss_input);
				if (test_object) {
				
					json::Object::Dictionary dictionary = test_object->dictionary["contents"]->as_object();
					if (dictionary["name"]->as_string() != "libkerr") { passes = false; }
					else if (dictionary["age"]->as_integer() != 1) { passes = false; }
					else if (dictionary["version"]->as_real() != 0.01) { passes = false; }
					delete test_object;
				}
				else {
					passes = false;
					if (verbose) {
						std::cout << "  failed to parse object for some reason\n";
					}
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  object test 3 - " << (passes ? "passes" : "fails") << "\n";
		}

		return passes;
	}

	bool test_json_array(bool verbose) {
		bool passes = true;
		json::Parser parser;

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "[ \"weight\" , 145 ]";
				json::Array *test_array = parser.parse_array(ss_input);
				if (test_array) {
					json::Array::ValueVector values = test_array->as_array();
					if (values.size() != 2) {
						passes = false;
						if (verbose) {
							std::cout << "array test 1 - expected 2 elements\n";
						}
					}
					else if (values[0]->as_string() != "weight") {
						passes = false;
						if (verbose) {
							std::cout << "array test 1 - expected element[0] = 'weight', got " 
								<< values[0]->as_string() << "\n";
						}
					}
					else if (values[1]->as_integer() != 145) {
						passes = false;
						if (verbose) {
							std::cout << "array test 1 - expected element[1] = 145, got " 
								<< values[1]->as_integer() << "\n";
						}
					}
					delete test_array;
				}
				else {
					passes = false;
					if (verbose) {
						std::cout << "  failed to parse object for some reason\n";
					}
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  array test 1 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "[]";
				json::Array *test_array = parser.parse_array(ss_input);
				if (test_array) {
					json::Array::ValueVector values = test_array->as_array();
					if (values.size() != 0) {
						passes = false;
						if (verbose) {
							std::cout << "array test 1 - expected 0 elements\n";
						}
					}
					delete test_array;
				}
				else {
					passes = false;
					if (verbose) {
						std::cout << "  failed to parse array for some reason\n";
					}
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  array test 2 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "[{\"id\": 1, \"location\": [0,0,0]}, {\"id\": 2, \"location\": [1,0,0]}]";
				json::Array *test_array = parser.parse_array(ss_input);
				if (test_array) {
					json::Array::ValueVector values = test_array->as_array();
					if (values.size() != 2) {
						passes = false;
						if (verbose) {
							std::cout << "array test 3 - expected 2 elements\n";
						}
					}
					else {
						for (int i = 0; passes && i < 2; i++) {
							json::Object::Dictionary dict = values[i]->as_object();
							if (dict["id"]->as_integer() != (i+1)) {
								passes = false;
							}
							else {
								json::Array::ValueVector location = dict["location"]->as_array();
								if (!(location[0]->as_integer() == i && location[1]->as_integer() == 0 && 
									location[2]->as_integer() == 0)) {
									passes = false;
								}
							}
						}
					}
					delete test_array;
				}
				else {
					passes = false;
					if (verbose) {
						std::cout << "  failed to parse array for some reason\n";
					}
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  array test 3 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input;
				ss_input << "[{\"id\":1, \"location\" : [ 0, 0,0]},{\"id\":2,\"location\":[1,0,0]}]";
				json::Array *test_array = parser.parse_array(ss_input);
				if (test_array) {
					json::Array::ValueVector values = test_array->as_array();
					if (values.size() != 2) {
						passes = false;
						if (verbose) {
							std::cout << "array test 4 - expected 2 elements\n";
						}
					}
					else {
						for (int i = 0; passes && i < 2; i++) {
							json::Object::Dictionary dict = values[i]->as_object();
							if (dict["id"]->as_integer() != (i+1)) {
								passes = false;
								if (verbose) {
									std::cout << "array test 4 - id test failed for object " << i << "\n";
								}
							}
							else {
								json::Array::ValueVector location = dict["location"]->as_array();
								if (!(location[0]->as_integer() == i && location[1]->as_integer() == 0 && 
									location[2]->as_integer() == 0)) {
									passes = false;
									if (verbose) {
										std::cout << "array test 4 - location test failed for object " << i << "\n";
									}
								}
							}
						}
					}
					delete test_array;
				}
				else {
					passes = false;
					if (verbose) {
						std::cout << "  failed to parse array for some reason\n";
					}
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}
		if (verbose) {
			std::cout << "  array test 4 - " << (passes ? "passes" : "fails") << "\n";
		}

		return passes;
	}

	bool test_json_comprehensive(bool verbose) {
		bool passes = true;
		json::Parser parser;

		if (passes) {
			std::stringstream ss;

			// example taken from some online source
			ss <<
				"{ \"accounting\" : [\n"
				"		{ \"firstName\" : \"John\",\n"
				"		\"lastName\"  : \"Doe\",\n"
				"		\"age\"       : 23 },\n"
				"		{ \"firstName\" : \"Mary\",\n"
				"		\"lastName\"  : \"Smith\",\n"
				"		\"age\"       : 32 }\n"
				"	],\n"                              
				"	\"sales\"       : [\n"
				"		{ \"firstName\" : \"Sally\",\n"
				"		\"lastName\"  : \"Green\",\n"
				"		\"age\"       : 27 },\n"
				"		{ \"firstName\" : \"Jim\",\n"
				"		\"lastName\"  : \"Galley\",\n"
				"		\"age\"       : 41 }\n"
				"	]\n"
				"}";
			const char *firstNames[] = {
				"John", "Mary", "Sally", "Jim", 0
			};
			const char *lastNames[] = {
				"Doe", "Smith", "Green", "Galley", 0
			};
			int ages[] = {
				23, 32, 27, 41, 0
			};
			try {
				json::Object *test_object = parser.parse_object(ss);
				json::Object::Dictionary dict = test_object->as_object();
		
				json::Array::ValueVector accounting = dict["accounting"]->as_array();
				json::Array::ValueVector sales = dict["sales"]->as_array();

				for (int i = 0; passes && i < 2; i++) {
					json::Object::Dictionary person = accounting[i]->as_object();
					if (person["firstName"]->as_string() != std::string(firstNames[i])) {
						passes = false;
					}
					else if (person["lastName"]->as_string() != std::string(lastNames[i])) {
						passes = false;
					}
					else if (person["age"]->as_integer() != ages[i]) {
						passes = false;
					}
				}
				for (int i = 0; passes && i < 2; i++) {
					json::Object::Dictionary person = sales[i]->as_object();
					if (person["firstName"]->as_string() != std::string(firstNames[i+2])) {
						passes = false;
					}
					else if (person["lastName"]->as_string() != std::string(lastNames[i+2])) {
						passes = false;
					}
					else if (person["age"]->as_integer() != ages[i+2]) {
						passes = false;
					}
				}

				if (passes) {
					std::stringstream emitted_output;
					json::Emitter emitter;
					emitter.use_tabs = false;
					emitter.emit_pretty(emitted_output, test_object);

					json::Object *reparsed_object = parser.parse_object(emitted_output);
					json::Object::Dictionary dict = reparsed_object->as_object();
		
					json::Array::ValueVector accounting = dict["accounting"]->as_array();
					json::Array::ValueVector sales = dict["sales"]->as_array();

					for (int i = 0; passes && i < 2; i++) {
						json::Object::Dictionary person = accounting[i]->as_object();
						if (person["firstName"]->as_string() != std::string(firstNames[i])) {
							passes = false;
						}
						else if (person["lastName"]->as_string() != std::string(lastNames[i])) {
							passes = false;
						}
						else if (person["age"]->as_integer() != ages[i]) {
							passes = false;
						}
					}
					for (int i = 0; passes && i < 2; i++) {
						json::Object::Dictionary person = sales[i]->as_object();
						if (person["firstName"]->as_string() != std::string(firstNames[i+2])) {
							passes = false;
						}
						else if (person["lastName"]->as_string() != std::string(lastNames[i+2])) {
							passes = false;
						}
						else if (person["age"]->as_integer() != ages[i+2]) {
							passes = false;
						}
					}

					delete reparsed_object;
				}
				delete test_object;
			}
			catch (hydrazine::Exception &exp) {
				std::cout << exp.what() << std::endl;
				std::cout << "\ntesting with:\n" << ss.str() << "\n\n";
				passes = false;
			}
			if (verbose) {
				std::cout << "  comprehensive test 1 - " << (passes ? "passes" : "fails") << "\n";
			}
		}

		return passes;
	}

	// test_json.cpp
	bool test_json_identifier(bool verbose) {
		bool passes = true;
		json::Parser parser;

		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "identifier";
				json::String *test_string = parser.parse_identifier(ss_input_string);
				if (test_string) {
					if (test_string->value_string != "identifier") {
						passes = false;
					}
					delete test_string;
				}
				else {
					passes = false;
				}
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}

		if (verbose) {
			std::cout << "  string test 4 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "[andrew, robert, kerr]";
				json::Array *seq = parser.parse_array(ss_input_string);
				if (seq->sequence.size() == 3) {
					const char *fixture[] = {"andrew", "robert", "kerr"};
					for (int i = 0; i < 3; i++) {
						if (std::string(fixture[i]) != seq->sequence[i]->as_string()) {
							passes = false;
						}
					}
				}
				else {
					passes = false;
				}
				delete seq;
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}

		if (verbose) {
			std::cout << "  string test 5 - " << (passes ? "passes" : "fails") << "\n";
		}

		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "[id3ntifier,identifier0,_123]";
				json::Array *seq = parser.parse_array(ss_input_string);
				if (seq->sequence.size() == 3) {
					const char *fixture[] = {"id3ntifier", "identifier0", "_123"};
					for (int i = 0; i < 3; i++) {
						if (std::string(fixture[i]) != seq->sequence[i]->as_string()) {
							passes = false;
							break;
						}
					}
				}
				else {
					passes = false;
				}
				delete seq;
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}

		if (verbose) {
			std::cout << "  string test 6 - " << (passes ? "passes" : "fails") << "\n";
		}
	
		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "{key0:id3ntifier,key1:identifier0,key2:_123}";
				json::Object *obj = parser.parse_object(ss_input_string);
				for (json::Object::Dictionary::iterator it = obj->begin(); it != obj->end(); ++it) {
					passes = false;
					if (it->first == "key0" && it->second->as_string() == "id3ntifier") {
						passes = true;
					}
					else if (it->first == "key1" && it->second->as_string() == "identifier0") {
						passes = true;
					}
					else if (it->first == "key2" && it->second->as_string() == "_123") {
						passes = true;
					}
				}
				delete obj;
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}

		if (verbose) {
			std::cout << "  string test 7 - " << (passes ? "passes" : "fails") << "\n";
		}

	
		try {
			if (passes) {
				std::stringstream ss_input_string;
				ss_input_string << "{key0 : [0, 1, 2], key1 : \"comment field\", key2 : null}";
				json::Object *obj = parser.parse_object(ss_input_string);

				delete obj;
			}
		}
		catch (hydrazine::Exception exp) {
			std::cout << exp.what() << std::endl;
			passes = false;
		}

		if (verbose) {
			std::cout << "  string test 8 - " << (passes ? "passes" : "fails") << "\n";
		}
		return passes;
	}

	bool test_json_visitor(bool verbose) {
		bool passes = true;
		json::Parser parser;

		try {
			std::stringstream ss;

			// without quotes, and with weird spacing
			ss << "{objects:[{id:1, location : [ 0, 0,0]},{id:2,location:[1.25,0.25,4]}]}";
		
			json::Object *object = parser.parse_object(ss);
			json::Visitor visitor(object);
		
			json::Visitor obj0 = visitor["objects"][0];
			json::Visitor obj1 = visitor["objects"][1];
		
			if ((int)(obj0["id"]) != 1 || (double)(obj0["location"][0]) != 0) {
				if (verbose) {
					std::cout << "visitor test 0 - failed for obj0\n";
				}
				passes = false;
			}
		
			if ((int)(obj1["id"]) != 2 || (double)(obj1["location"][0]) != 1.25 ||
				(double)(obj1["location"][1]) != 0.25 || (double)(obj1["location"][2]) != 4) {
				if (verbose) {
					std::cout << "visitor test 0 - failed for obj1\n";
				}
				passes = false;			
			}
		
			delete object;
		}
		catch (hydrazine::Exception &exp) {
			passes = false;
			std::cout << "parse failed for visitor test 0\n";
			std::cout << exp.what() << std::endl;
		}
	
		try {
			std::stringstream ss;

			// example taken from some online source
			ss <<
				"{ \"accounting\" : [\n"
				"		{ \"firstName\" : \"John\",\n"
				"		\"lastName\"  : \"Doe\",\n"
				"		\"age\"       : 23 },\n"
				"		{ \"firstName\" : \"Mary\",\n"
				"		\"lastName\"  : \"Smith\",\n"
				"		\"age\"       : 32 }\n"
				"	],\n"                              
				"	\"sales\"       : [\n"
				"		{ \"firstName\" : \"Sally\",\n"
				"		\"lastName\"  : \"Green\",\n"
				"		\"age\"       : 27 },\n"
				"		{ \"firstName\" : \"Jim\",\n"
				"		\"lastName\"  : \"Galley\",\n"
				"		\"age\"       : 41 }\n"
				"	]\n"
				"}";
			const char *firstNames[] = {
				"John", "Mary", "Sally", "Jim", 0
			};
			const char *lastNames[] = {
				"Doe", "Smith", "Green", "Galley", 0
			};
			int ages[] = {
				23, 32, 27, 41, 0
			};
		
			json::Object *object = parser.parse_object(ss);
			json::Visitor visitor(object);
		
			int j = 0;
			const char *departments[] = { "accounting", "sales" };
			for (int as = 0; as < 2; as++) {
				const char *department = departments[as];
				for (int p = 0; p < 2; p++, j++) {
					if ((std::string)(visitor[department][p]["firstName"]) != firstNames[j] || 
						(std::string)(visitor[department][p]["lastName"]) != lastNames[j] || 
						(int)(visitor[department][p]["age"]) != ages[j]) {
						if (verbose) {
							std::cout << "visitor test 1 - failed for accounting[0]\n";
						}
						passes = false;
					}
				}
			}
		
			delete object;
		}
		catch (hydrazine::Exception &exp) {
			passes = false;
			std::cout << "parse failed for visitor test 0\n";
			std::cout << exp.what() << std::endl;
		}
	
		if (verbose) {
			std::cout << "  visitor test - " << (passes ? "passes" : "fails") << "\n";
		}
	
		return passes;
	}

	bool test_json(bool verbose) {
		bool passes = true;

		if (verbose) {
			std::cout << "test_json - entered.." << std::endl;
		}

		passes = test_json_string(verbose);

		passes = (passes && test_json_number(verbose));
		passes = (passes && test_json_object(verbose));
		passes = (passes && test_json_array(verbose));
		passes = (passes && test_json_identifier(verbose));
		passes = (passes && test_json_comprehensive(verbose));
		passes = (passes && test_json_visitor(verbose));

		if (verbose) {
			if (passes) {
				std::cout << "test_json - Success" << std::endl;
			}
			else {
				std::cout << "test_json - FAILED" << std::endl;
			}
		}
		return passes;
	}

	TestJson() {

		name = "TestJson";
		
		description = "Tests the JSON parser by parsing a number of valid JSON strings and comparing results";
	}

private:

	bool doTest() {
		return test_json(verbose);
	}
};

int main(int argc, char *argv[]) {
	hydrazine::ArgumentParser parser( argc, argv );

	TestJson test;

	parser.description( test.testDescription() );
	parser.parse( "-v", test.verbose, false, 
		"Print out information after the test is over." );

	test.test();

	return test.passed();
}

