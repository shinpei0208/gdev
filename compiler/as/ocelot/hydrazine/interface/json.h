/*!
	\file json.h

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief defines a JSON parser and emitter

	\date 27 Oct 2009
*/

#ifndef HYDRAZINE_JSON_H_INCLUDED
#define HYDRAZINE_JSON_H_INCLUDED

#include <vector>
#include <map>
#include <string>
#include <fstream>

namespace hydrazine {
namespace json {

	/*!
		Base value type - True, False, and Null do not have derived classes
	*/
	class Value {
	public:
		enum Type {
			Object,
			Array,
			DenseArray,
			Number,
			String,
			True,
			False,
			Null,
			Type_invalid
		};

	public:
		Value();
		Value(Type type);
		virtual ~Value();

		/*!
			makes a deep copy of the value
		*/
		virtual Value *clone() const;

	public:
		/*
			accessors perform cast if possible or throw an exception
		*/

		//! returns an integer if the value is a Number and an integer
		unsigned long long int as_integer() const;

		//! returns a double if the value is a Number and a real
		double as_real() const;

		//! returns a double if the value is a Number; casts to double if an integer
		double as_number() const;

		//! returns a string representation if the value is a String
		const std::string& as_string() const;

		//! returns a vector of values if the value is an array
		const std::vector< Value *>& as_array() const;
		
		//! returns a vector of integers if the value is a dense array
		const std::vector< int >& as_dense_array() const;
		
		//! returns a dictionary of values if the value is an object
		const std::map< std::string, Value *>& as_object() const;
		
		//! returns true or false if the value is true or false respectively
		bool as_boolean() const;

		//! returns true if the value is null, false if not - doesn't throw exception
		bool is_null() const;

	public:

		Type type;
	};

	/*!
		Numeric scalar value
	*/
	class Number : public Value {
	public:
		enum NumberType {
			Integer,
			Real,
			Number_invalid
		};

	public:

		Number();
		Number(double real_value);
		Number(int int_value);
		virtual ~Number();

		virtual Value *clone() const;

	public:

		NumberType number_type;

		double value_real;

		unsigned long long int value_integer;
	};

	/*!
		Represents an ordered list of values
	*/
	class Array : public Value {
	public:

		typedef std::vector< Value * > ValueVector;

		typedef ValueVector::iterator iterator;
		typedef ValueVector::const_iterator const_iterator;
		
	public:
		Array();
		Array(const ValueVector &values);
		virtual ~Array();

		iterator begin();
		const_iterator begin() const;
		iterator end();
		const_iterator end() const;

		size_t size() const;

		virtual Value *clone() const;

	public:

		ValueVector sequence;
	};

	/*!
		Represents an ordered list of integer values
	*/
	class DenseArray : public Value {
	public:

		typedef std::vector< int > IntVector;

		typedef IntVector::iterator iterator;
		typedef IntVector::const_iterator const_iterator;
		
	public:
		DenseArray();
		DenseArray(const IntVector &values);
		virtual ~DenseArray();

		iterator begin();
		const_iterator begin() const;
		iterator end();
		const_iterator end() const;

		Value* number(int index);

		virtual Value *clone() const;

	public:
		IntVector sequence;

	private:
		json::Number _temp;
	};

	/*!
			Represents a string scalar value
	*/
	class String : public Value {
	public:
		String();
		String(const std::string &string_value);
		virtual ~String();

		virtual Value *clone() const;

	public:

		std::string value_string;
	};

	/*!
		A dictionary mapping unique strings onto values
	*/
	class Object : public Value {
	public:
		typedef std::pair< std::string, Value *> KeyValuePair;
		typedef std::map< std::string, Value *> Dictionary;
		
		typedef Dictionary::iterator iterator;
		typedef Dictionary::const_iterator const_iterator;
		
	public:
		Object();
		Object(const Dictionary &object);
		virtual ~Object();

		iterator begin();
		const_iterator begin() const;
		iterator end();
		const_iterator end() const;

		virtual Value *clone() const;

	public:
		Dictionary dictionary;
	};

	/*!
		Parses an istream into a JSON array
	*/
	class Parser {
	public:
		Parser();
		~Parser();

		/*!
			parses an istream into a JSON array
		*/
		Array *parse(std::istream &input);

		int line_number;

	public:

		int get_non_whitespace_char(std::istream &input);
		int get_char(std::istream &input);
		void putback(std::istream &input, int ch);

		Value *parse_value(std::istream &input);
		Value *parse_array(std::istream &input);
		Object *parse_object(std::istream &input);
		Number *parse_number(std::istream &input);
		String *parse_string(std::istream &input);
		String *parse_identifier(std::istream &input);
	};

	/*!
		Emits a JSON object to an ostream
	*/
	class Emitter {
	public:
		Emitter();
		~Emitter();

		/*!
			Emits a JSON value to an ostream
		*/
		std::ostream & emit(std::ostream &output, Value *value);

		void emit_pretty(std::ostream &output, const Value *value, int indent_level = 0);
		void emit_compact(std::ostream &output, const Value *value);

		void emit_array_pretty(std::ostream &output, const Array *object, int indent_level=0);
		void emit_dense_array_pretty(std::ostream &output, const DenseArray *object, int indent_level=0);
		void emit_object_pretty(std::ostream &output, const Object *object, int indent_level=0);
		void emit_number(std::ostream &output, const Number *number);
		void emit_string(std::ostream &output, const std::string &str);

		void emit_indents(std::ostream &output, int indents);

	public:

		/*!
			If true, indents with tab (\t) character. If false, indents with space (0x20)
		*/
		bool use_tabs;

		/*!
			Number of space (0x20) or tab (\t) characters to indent nested objects
		*/
		int indent_size;

	};
	
	/*!
		Class wrapping value structure enabling programmer-friendly access. Methods throw
		common::exception if accesses are invalid
	*/
	class Visitor {
	public:
		Visitor();
		~Visitor();
		
		Visitor(Value *value);

		Value *find(const std::string & obj) const;
		
		//! assuming value is an Object, returns a Visitor for the named value
		Visitor operator[](const char *) const;
		
		//! assuming value is an Array, returns a Visitor for the indexed value
		Visitor operator[](int ) const;
		
		//! returns true if value is Null
		bool is_null() const;
		
		//! casts value to boolean, assuming it is either True or False
		operator bool() const;
		
		//! casts value to an integer, assuming it is a Number
		operator int() const;
		
		//! casts value to a double, assuming it is a Number
		operator double() const;
		
		//! casts value to a string, assuming it is a String
		operator std::string() const;
		
		Array::iterator begin_array();
		Array::const_iterator begin_array() const;
		Array::iterator end_array();
		Array::const_iterator end_array() const;
		size_t size_array() const;
		
		Object::iterator begin_object();
		Object::const_iterator begin_object() const;
		Object::iterator end_object();
		Object::const_iterator end_object() const;
		

		template <typename T> T parse(const char *key, T defaultValue) const {
			if (find(key)) {
				return (T)(*this)[key];
			}
			return defaultValue;
		}
		
	public:
	
		//! value mapped to visitor
		Value *value;
	};
}
}

#endif
