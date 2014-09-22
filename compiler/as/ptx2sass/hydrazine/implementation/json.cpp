/*!
	\file json.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief defines a JSON parser and emitter

	\date 27 Oct 2009
*/

#include <hydrazine/interface/json.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

#include <math.h>
#include <sstream>
#include <deque>
#include <stdexcept>
#include <cstring>

#define EXCEPTION(message) std::runtime_error(message)

#define throw_EXCEPTION_line(line, msg, stream) { std::stringstream ss; \
	char buffer[32]; std::memset(buffer, 0, 32); stream.getline(buffer, 31); \
	ss << "line " << line << ": " << msg << "\n before: " << buffer; throw std::runtime_error(ss.str()); }

////////////////////////////////////////////////////////////////////////////////
namespace hydrazine {

json::Value::Value(): type(Null) {

}
json::Value::Value(Type _type): type(_type) {
	
}

json::Value::~Value() {

}

json::Value *json::Value::clone() const {
	return 0;
}

unsigned long long int json::Value::as_integer() const {
	if (type == Value::Number) {
		const json::Number *number = static_cast<const json::Number *>(this);
		if (number->number_type == Number::Integer) {
			return number->value_integer;
		}
	}
	throw EXCEPTION("Invalid cast");
}

double json::Value::as_real() const {
	if (type == Value::Number) {
		const json::Number *number = static_cast<const json::Number *>(this);
		if (number->number_type == Number::Real) {
			return number->value_real;
		}
	}
	throw EXCEPTION("Invalid cast");
}

double json::Value::as_number() const {
	if (type == Value::Number) {
		const json::Number *number = static_cast<const json::Number *>(this);
		if (number->number_type == Number::Real) {
			return number->value_real;
		}
		else if (number->number_type == Number::Integer) {
			return (double)number->value_integer;
		}
	}
	throw EXCEPTION("Invalid cast");
}

const std::string& json::Value::as_string() const {
	if (type == Value::String) {
		const json::String *element = static_cast<const json::String *>(this);
		return element->value_string;
	}
	throw EXCEPTION("Invalid cast");
}

const std::vector< json::Value *>& json::Value::as_array() const {
	if (type == Value::Array) {
		const json::Array *element = static_cast<const json::Array *>(this);
		return element->sequence;
	}
	throw EXCEPTION("Invalid cast");
}

const std::vector< int >& json::Value::as_dense_array() const {
	if (type == Value::DenseArray) {
		const json::DenseArray *element = static_cast<const json::DenseArray *>(this);
		return element->sequence;
	}
	throw EXCEPTION("Invalid cast");
}

const std::map< std::string, json::Value *>& json::Value::as_object() const {
	if (type == Value::Object) {
		const json::Object *element = static_cast<const json::Object *>(this);
		return element->dictionary;
	}
	throw EXCEPTION("Invalid cast");
}


//! returns true or false if the value is true or false respectively
bool json::Value::as_boolean() const {
	if (type == Value::True) {
		return true;
	}
	else if (type == Value::False) {
		return false;
	}
	throw EXCEPTION("Invalid cast");
}

//! returns true if the value is null, false if not - doesn't throw exception
bool json::Value::is_null() const {
	return (type == Value::Null);
}

////////////////////////////////////////////////////////////////////////////////

json::Number::Number(): Value(Value::Number) {

}

json::Number::Number(double real_value): Value(Value::Number), number_type(Real),
										value_real(real_value), value_integer(0) {

}

json::Number::Number(int int_value): Value(Value::Number), number_type(Integer), 
										 value_real(0), value_integer(int_value)  {

}

json::Number::~Number() {

}

json::Value *json::Number::clone() const {
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

json::Array::Array(): Value(Value::Array) {

}

json::Array::Array(const json::Array::ValueVector &values): Value(Value::Array), 
	sequence(values) {

}

json::Array::~Array() {
	for (ValueVector::iterator val_it = begin(); val_it != end(); ++val_it) {
		delete *val_it;
	}
}

json::Array::ValueVector::iterator json::Array::begin() {
	return sequence.begin();
}

json::Array::ValueVector::const_iterator json::Array::begin() const {
	return sequence.begin();
}

json::Array::ValueVector::iterator json::Array::end() {
	return sequence.end();
}

json::Array::ValueVector::const_iterator json::Array::end() const {
	return sequence.end();
}

size_t json::Array::size() const {
	return sequence.size();
}

json::Value *json::Array::clone() const {
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

json::DenseArray::DenseArray(): Value(Value::DenseArray) {

}

json::DenseArray::DenseArray(const json::DenseArray::IntVector &values): Value(Value::DenseArray), 
	sequence(values) {

}

json::DenseArray::~DenseArray() {

}

json::DenseArray::IntVector::iterator json::DenseArray::begin() {
	return sequence.begin();
}

json::DenseArray::IntVector::const_iterator json::DenseArray::begin() const {
	return sequence.begin();
}

json::DenseArray::IntVector::iterator json::DenseArray::end() {
	return sequence.end();
}

json::DenseArray::IntVector::const_iterator json::DenseArray::end() const {
	return sequence.end();
}

json::Value* json::DenseArray::number(int index) {
	_temp = json::Number(sequence[index]);
	
	return &_temp;
}


json::Value *json::DenseArray::clone() const {
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

json::String::String(): Value(Value::String) {

}

json::String::String(const std::string &string_value): Value(Value::String), 
	value_string(string_value) {

}

json::String::~String() {
}

json::Value *json::String::clone() const {
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

json::Object::Object(): Value(Value::Object) {

}

json::Object::Object(const json::Object::Dictionary &object): Value(Value::Object), 
	dictionary(object) {

}

json::Object::~Object() {
	for (Dictionary::iterator d_it = dictionary.begin(); d_it != dictionary.end(); ++d_it) {
		delete d_it->second;
	}
	dictionary.clear();
}

json::Object::Dictionary::iterator json::Object::begin() {
	return dictionary.begin();
}

json::Object::Dictionary::const_iterator json::Object::begin() const {
	return dictionary.begin();
}

json::Object::Dictionary::iterator json::Object::end() {
	return dictionary.end();
}

json::Object::Dictionary::const_iterator json::Object::end() const {
	return dictionary.end();
}

json::Value *json::Object::clone() const {
	return 0;
}

////////////////////////////////////////////////////////////////////////////////

json::Parser::Parser(): line_number(0) {

}

json::Parser::~Parser() {

}

json::Array *json::Parser::parse(std::istream &input) {
	return 0;
}

static bool is_whitespace_char(int ch) {
	if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
		return true;
	}
	return false;
}

int json::Parser::get_non_whitespace_char(std::istream &input) {
	int ch;
	bool reading = true;
	bool in_comment = false;
	
	do {
		ch = get_char(input);
		if (in_comment) {
			if(ch == '\n' || ch == '\r')
			{
				in_comment = false;
			}
		}
		else if (ch == '#') {
			in_comment = true;
		}
		else if (!is_whitespace_char(ch)) {
			reading = false;
		}
	} while (reading);
	return ch;
}

int json::Parser::get_char(std::istream &input) {
	int ch = input.get();
	if (ch == '\n') {
		++line_number;
	}
	return ch;
}

void json::Parser::putback(std::istream &input, int ch) {
	if (ch == '\n') {
		--line_number;
	}
	input.putback(ch);
}

json::Value *json::Parser::parse_value(std::istream &input) {
	int ch = get_non_whitespace_char(input);
	switch (ch) {
		case '{':
			//std::cout << "parse_value - parsing object..\n";
			putback(input, ch);
			return parse_object(input);
			break;

		case '[':
			//std::cout << "parse_value - parsing array..\n";
			putback(input, ch);
			return parse_array(input);
			break;

		case '"':
			//std::cout << "parse_value - parsing string..\n";
			putback(input, ch);
			return parse_string(input);
			break;

		default:
			putback(input, ch);
			if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_')) {
				json::String *identifier = parse_identifier(input);
				if (identifier->as_string() == "true" || identifier->as_string() == "True") {
					delete identifier;
					return new Value(Value::True);
				}
				else if (identifier->as_string() == "false" || identifier->as_string() == "False") {
					delete identifier;
					return new Value(Value::False);
				}
				else if (identifier->as_string() == "null") {
					delete identifier;
					return new Value(Value::Null);
				}
				return identifier;
			}
			else {
				return parse_number(input);
			}
	}
	return 0;
}

json::Value *json::Parser::parse_array(std::istream &input) {
	typedef std::deque<int> FastIntVector;
	
	FastIntVector denseSequence;
	
	Array::ValueVector sequence;
	enum States {
		initial,
		open_bracket,
		close_bracket,
		value,
		comma,
		exit,
		invalid
	};
	States state = initial;
	bool isDense = true;
	
	do {
		char ch = get_non_whitespace_char(input);
		switch (state) {
			case initial: 
				{
					if (ch == '[') {
						state = value;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_array() - unexpected character '" << ch << "' expected '['", input);
					}
				}
				break;
			case value: 
				{
					if (ch == ']') {
						state = exit;
					}
					else {
						putback(input, ch);
						json::Value *active_value = parse_value(input);
						if (active_value) {
							if (active_value->type == Value::Number &&
								static_cast<Number*>(active_value)->number_type == Number::Integer) {
								
								denseSequence.push_back(active_value->as_integer());
								
								delete active_value;
								
								ch = get_non_whitespace_char(input);
								if (ch == ']') {
									state = exit;
								}
								else if (ch == ',') {
									state = value;
								}
								else {
									throw_EXCEPTION_line(line_number, "json::Parser::parse_array() - unexpected character'" << ch << "' expected ','", input);
								}
							}
							else {
								if (isDense) {
									isDense = false;
									
									for (auto i = denseSequence.begin();
										i != denseSequence.end(); ++i) {
										sequence.push_back(new Number(*i));
									}
								}
								
								sequence.push_back(active_value);
								ch = get_non_whitespace_char(input);
								if (ch == ']') {
									state = exit;
								}
								else if (ch == ',') {
									state = value;
								}
								else {
									throw_EXCEPTION_line(line_number, "json::Parser::parse_array() - unexpected character; expected ','", input);
								}
							}
						}
						else {
							throw_EXCEPTION_line(line_number, "json::Parser::parse_array() - failed to parse value", input);
						}
					}
				}
				break;
			case exit: 
			default: {
				}
				break;
		}
	} while (state != exit);
	
	if (isDense && !denseSequence.empty()) {
		return new json::DenseArray(json::DenseArray::IntVector(
			denseSequence.begin(), denseSequence.end()));
	}
	else {
		return new json::Array(sequence);
	}
}

json::Object *json::Parser::parse_object(std::istream &input) {
	enum States {
		initial,
		open_brace,
		string,
		identifier,
		colon,
		value,
		comma,
		close_brace,
		exit,
		invalid
	};
	String *active_string = 0;
	Value *active_value = 0;
	Object::Dictionary dictionary;

	States state = initial;
	do {
		int ch = get_non_whitespace_char(input);
		switch (state) {
			case initial:
				{
					if (ch == '{') {
						state = open_brace;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - unexpected character in object", input);
					}
				}
				break;

			case open_brace:
				{
					if (ch == '}') {
						state = close_brace;
					}
					else if (ch == '"') {
						putback(input, ch);
						state = string;
					}
					else if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_')) {
						state = identifier;
						putback(input, ch);
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - unexpected key character found", input);
					}
				}
				break;

			case string:
				{
					putback(input, ch);
					active_string = parse_string(input);
					if (active_string) {
						ch = get_non_whitespace_char(input);
						if (ch == ':') {
							state = value;
							break;
						}
						else {
							throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - expected colon after key string", input);
						}
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - failed to parse key string", input);
					}
				}
				break;

			case identifier:
				{
					putback(input, ch);
					active_string = parse_identifier(input);
					if (active_string) {
						ch = get_non_whitespace_char(input);
						if (ch == ':') {
							state = value;
							break;
						}
						else {
							throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - expected colon after key string", input);
						}
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - failed to parse key string", input);
					}
				}
				break;

			case colon:
				{
					putback(input, ch);
					state = value;
				}
				break;

			case value:
				{
					putback(input, ch);
					active_value = parse_value(input);
					if (active_value) {
						assert(dictionary.count(active_string->value_string) == 0);
						dictionary[active_string->value_string] = active_value;
						delete active_string;
						ch = get_non_whitespace_char(input);
						if (ch == ',') {
							state = open_brace;
						}
						else if (ch == '}') {
							state = close_brace;
						}
						else {
							throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - unexpected char after value", input);
						}
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - failed to parse value", input);
					}
				}
				break;

			case comma:
				{
					if (ch == '"') {
						putback(input, ch);
						state = string;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_object() - unexpected char after comma", input);
					}
				}
				break;

			case close_brace:
				{
					putback(input, ch);
					state = exit;
				}
				break;

			case exit:
			case invalid:
			default:
				break;
		};
	} while (state != exit);

	return new Object(dictionary);
}

json::Number *json::Parser::parse_number(std::istream &input) {
	enum States {
		initial,
		negativeA,
		leading_zero,
		digit1_9,
		digitA,
		decimal,
		digitB,
		exponent,
		positiveExponent,
		negativeExponent,
		digitC,
		exit,
		invalid
	};

	bool positive = true;
	bool exponentPositive = true;

	unsigned long long int part_whole = 0, part_decimal = 0, part_exponent = 0;
	int digits_whole = 0, digits_decimal = 0, digits_exponent = 0;

	States state = initial;
	do {
		int ch = get_char(input);
		switch (state) {
			case initial:
				{
					if (ch == '-') {
						state = negativeA;
					}
					else if (ch == '0') {
						state = leading_zero;
						digits_whole ++;
					}
					else if (ch >= '1' && ch <= '9') {
						state = digit1_9;
						part_whole = (int)(ch - '0');
						digits_whole ++;
					}
					else if (is_whitespace_char(ch)) {
						state = initial;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_number() [initial] - unexpected character found", input);
					}
				}
				break;
			case negativeA:
				{
					positive = false;
					if (ch == '0') {
						state = leading_zero;
						digits_whole ++;
					}
					else if (ch >= '1' && ch <= '9') {
						state = digit1_9;
						part_whole = (int)(ch - '0');
						digits_whole ++;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_number() [negativeA] - unexpected character found", input);
					}
				}
				break;
			case leading_zero:
				{
					if (ch == '.') {
						state = decimal;
					}
					else {
						putback(input, ch);
						state = exit;
					}
				}
				break;
			case digit1_9:
				{
					if (ch >= '0' && ch <= '9') {
						part_whole = part_whole * 10 + (int)(ch - '0');
						digits_whole ++;
						state = digitA;
					}
					else if (ch == '.') {
						state = decimal;
					}
					else {
						input.putback(ch);
						state = exit;
					}
				}
				break;
			case digitA:
				{
					if (ch >= '0' && ch <= '9') {
						part_whole = part_whole * 10 + (int)(ch - '0');
						digits_whole ++;
						state = digitA;
					}
					else if (ch == '.') {
						state = decimal;
					}
					else if (ch == 'e' || ch == 'E') {
						state = exponent;
					}
					else {
						input.putback(ch);
						state = exit;
					}
				}
				break;
			case decimal:
				{
					if (ch >= '0' && ch <= '9') {
						state = digitB;
						part_decimal = (int)(ch - '0');
						digits_decimal ++;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_number() [decimal] - unexpected character found", input);
					}
				}
				break;
			case digitB:
				{
					if (ch >= '0' && ch <= '9') {
						state = digitB;
						part_decimal = part_decimal * 10 + (int)(ch - '0');
						digits_decimal ++;
					}
					else if (ch == 'e' || ch == 'E') {
						state = exponent;
					}
					else {
						putback(input, ch);
						state = exit;
					}
				}
				break;
			case exponent:
				{
					if (ch == '+') {
						state = positiveExponent;
					}
					else if (ch == '-') {
						state = negativeExponent;
					}
					else if (ch >= '0' && ch <= '9') {
						part_exponent = (int)(ch - '0');
						digits_exponent ++;
						state = digitC;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_number() [exponent] - unexpected character found", input);
					}
				}
				break;
			case positiveExponent:
				{
					if (ch >= '0' && ch <= '9') {
						part_exponent = (int)(ch - '0');
						digits_exponent ++;
						state = digitC;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_number() [positiveExponent] - unexpected character found", input);
					}
				}
				break;
			case negativeExponent:
				{
					exponentPositive = false;
					if (ch >= '0' && ch <= '9') {
						part_exponent = (int)(ch - '0');
						digits_exponent ++;
						state = digitC;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_number() [negativeExponent] - unexpected character found", input);
					}
				}
				break;
			case digitC:
				{
					if (ch >= '0' && ch <= '9') {
						state = digitC;
						part_exponent = part_exponent * 10 + (int)(ch - '0');
						digits_exponent ++;
					}
					else {
						putback(input, ch);
						state = exit;
					}
				}
				break;
			default:
				break;
		}
	} while (state != exit);

	Number *number = new Number;
	if (digits_exponent == 0 && digits_decimal == 0) {
		number->number_type = Number::Integer;
		number->value_integer = (positive ? part_whole : -part_whole);
		number->value_real = (double)number->value_integer;
	}
	else {
		double exponential = 1.0;
		number->number_type = Number::Real;
		number->value_real = (positive ? 1 : -1) * (
			(double)part_whole + (double)part_decimal / pow(10.0, digits_decimal));
		if (digits_exponent) {
			exponential = pow(10.0, (double)part_exponent);
			if (!exponentPositive) {
				exponential = 1.0 / exponential;
			}
			number->value_real *= exponential;
		}
		number->value_integer = (unsigned long long int)number->value_real;
	}

	return number;
}
/*
static bool is_unicode_character(int ch) {
	if (ch != '"') {
		return false;
	}
	if (ch != '\\') {
		return false;
	}
	return true;
}
*/
static char char_to_hex_digit(int ch) {
	if (ch >= '0' && ch <= '9') {
		return (char)(ch - '0');
	}
	else if (ch >= 'a' && ch <= 'f') {
		return 10 + (char)(ch - 'a');
	}
	else if (ch >= 'A' && ch <= 'F') {
		return 10 + (char)(ch - 'A');
	}
	return 0;
}

json::String *json::Parser::parse_identifier(std::istream &input) {
	enum States {
		initial,
		body_char,
		exit,
		invalid
	};
	std::stringstream ss;
	States state = initial;

	do {
		int ch = get_char(input);
		switch (state) {
			case initial:
				if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_')) {
					state = body_char;
					ss.write((const char *)&ch, (ch < 256 ? 1 : 2));
				}
				else {
					putback(input, ch);
					state = exit;
				}
				break;
			case body_char:
				if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_') ||
					(ch >= '0' && ch <= '9')) {
					state = body_char;
					ss.write((const char *)&ch, (ch < 256 ? 1 : 2));
				}
				else {
					putback(input, ch);
					state = exit;
				}
				break;
			default:
				throw_EXCEPTION_line(line_number, "json::Parser::parse_identifier() - invalid state", input);
		}
	} while (state != exit);
	return new String(ss.str());
}

json::String *json::Parser::parse_string(std::istream &input) {
	enum States {
		initial,
		leading_quote,
		trailing_quote,
		unicode_character,
		backslash,
		u0, u1, u2, u3,
		exit,
		invalid
	};

	std::stringstream ss;
	States state = initial;
	char hex[5];
	do {
		int ch = get_char(input);
		switch (state) {
			case initial:
				{
					if (ch == '"') {
						state = leading_quote;
					}
					else {
						throw_EXCEPTION_line(line_number, "json::Parser::parse_string() - unexpected character", input);
					}
				}
				break;

			case leading_quote:
				{
					if (ch == '"') {
						state = exit;
					}
					else if (ch == '\\') {
						state = backslash;
					}
					else {
						state = leading_quote;
						ss.write((const char *)&ch, (ch < 256 ? 1 : 2));
					}
				}
				break;

			case trailing_quote:
				{
					putback(input, ch);
					state = exit;
				}
				break;
				
			case backslash:
				{
					switch (ch) {
					case '"':
						ss << "\"";
						state = leading_quote;
						break;
					case '\\':
						ss << "\\";
						state = leading_quote;
						break;
					case '/':
						ss << "/";
						state = leading_quote;
						break;
					case 'b':
						ss << "\b";
						state = leading_quote;
						break;
					case 'f':
						ss << "\f";
						state = leading_quote;
						break;
					case 'n':
						ss << "\n";
						state = leading_quote;
						break;
					case 'r':
						ss << "\r";
						state = leading_quote;
						break;
					case 't':
						ss << "\t";
						state = leading_quote;
						break;
					case 'u':
						state = u0;
						break;
					}
				}
				break;

			case u0:
				{
					hex[0] = char_to_hex_digit(ch);
					state = u1;
				}
				break;

			case u1:
				{
					hex[1] = char_to_hex_digit(ch);
					state = u2;
				}
				break;

			case u2:
				{
					hex[2] = char_to_hex_digit(ch);
					state = u3;
				}
				break;

			case u3:
				{
					int hex_value = 0;
					hex[3] = char_to_hex_digit(ch);
					hex[4] = 0;
					state = leading_quote;
					hex[0] += (hex[1] << 8);
					hex[1] = (hex[3] << 8) + hex[1];
					hex_value = (int)hex[0] + ((int)hex[1] << 8);
					ss.write(hex, (hex_value > 127 ? 2 : 1));
				}
				break;

			case exit:
				{
				}
				break;

			default:
				break;
		}
	} while (state != exit);

	return new String(ss.str());
}

////////////////////////////////////////////////////////////////////////////////

json::Emitter::Emitter(): use_tabs(true), indent_size(1) {

}

json::Emitter::~Emitter() {

}

std::ostream & json::Emitter::emit(std::ostream &output, json::Value *value) {
	return output;
}

void json::Emitter::emit_number(std::ostream &output, const Number *number) {
	if (number->number_type == Number::Integer) {
		output << number->value_integer;
	}
	else {
		output << number->value_real;
	}
}

void json::Emitter::emit_string(std::ostream &output, const std::string &str) {
	output << "\"";
	for (size_t i = 0; i < str.size(); i++) {
		switch (str[i]) {
			case '\n':
				output << "\\n";
				break;
			case '\\':
				output << "\\\\";
				break;
			case '/':
				output << "\\/";
				break;
			case '\f':
				output << "\\f";
				break;
			case '\r':
				output << "\\r";
				break;
			case '\t':
				output << "\\t";
				break;
			default:
				output.put(str[i]);
				break;
		}
	}
	output << "\"";
}

void json::Emitter::emit_indents(std::ostream &output, int indents) {
	for (int i = 0; i < indents * indent_size; i++) {
		output << (use_tabs ? "\t" : " ");
	}
}

void json::Emitter::emit_object_pretty(std::ostream &output, const Object *object, int indents) {
	output << "{\n";
	int n = 0;

	for (json::Object::Dictionary::const_iterator key_it = object->dictionary.begin(); 
		key_it != object->dictionary.end(); ++key_it, ++n) {
		
		if (n) {
			output << ",\n";
		}

		emit_indents(output, indents+1);
		emit_string(output, key_it->first);
		output << ": ";
		emit_pretty(output, key_it->second, indents+1);
	}
	output << "\n";
	emit_indents(output, indents);
	output << "}";
}

void json::Emitter::emit_array_pretty(std::ostream &output, const Array *object, int indents) {
	output << "[\n";
	int n = 0;

	for (json::Array::ValueVector::const_iterator val_it = object->sequence.begin(); 
		val_it != object->sequence.end(); ++val_it, ++n) {
		if (n) {
			output << ",\n";
		}
		emit_indents(output, indents+1);
		emit_pretty(output, *val_it, indents+1);
	}
	output << "\n";
	emit_indents(output, indents);
	output << "]";
}

void json::Emitter::emit_dense_array_pretty(std::ostream &output, const DenseArray *object, int indents) {
	output << "[\n";
	int n = 0;

	output << std::hex;
	for (json::DenseArray::IntVector::const_iterator val_it = object->sequence.begin(); 
		val_it != object->sequence.end(); ++val_it, ++n) {
		if (n) {
			output << ",\n";
		}
		emit_indents(output, indents+1);
		output << *val_it;
	}
	output << std::dec << "\n";
	emit_indents(output, indents);
	output << "]";
}

void json::Emitter::emit_pretty(std::ostream &output, const json::Value *value, int indent_level) {
	switch (value->type) {
		case Value::Null:
			output << "null";
			break;

		case Value::True:
			output << "true";
			break;

		case Value::False:
			output << "false";
			break;

		case Value::Number:
			emit_number(output, static_cast<const Number*>(value));
			break;

		case Value::String:
			emit_string(output, static_cast<const String*>(value)->value_string);
			break;

		case Value::Object:
			emit_object_pretty(output, static_cast<const Object*>(value), indent_level);
			break;

		case Value::Array:
			emit_array_pretty(output, static_cast<const Array*>(value), indent_level);
			break;

		case Value::DenseArray:
			emit_dense_array_pretty(output, static_cast<const DenseArray*>(value), indent_level);
			break;

		default:
			break;
	}
}

void json::Emitter::emit_compact(std::ostream &output, const json::Value *value) {

}

////////////////////////////////////////////////////////////////////////////////

json::Visitor::Visitor(): value(0) { }
json::Visitor::~Visitor() { }

json::Visitor::Visitor(json::Value *_value): value(_value) { }

//! returns true if value is Null
bool json::Visitor::is_null() const {
	return !value || value->type == Value::Null;
}

json::Visitor json::Visitor::operator[](const char *key) const {
	if (value->type != Value::Object) {
		throw EXCEPTION("operator[](const std::string &) expects Visitor to wrap an Object");
	}
	Object *object = static_cast<Object*>(value);
	return Visitor(object->dictionary[key]);
}

//! assuming value is an Array, returns a Visitor for the indexed value
json::Visitor json::Visitor::operator[](int index) const {
	if (value->type != Value::Array && value->type != Value::DenseArray) {
		throw EXCEPTION("operator[](int) expects Visitor to wrap an Array");
	}

	if (value->type == Value::DenseArray) {
		DenseArray *denseArray = static_cast<DenseArray*>(value);
		
		return Visitor(denseArray->number(index));
	}

	Array *array = static_cast<Array*>(value);
	return Visitor(array->sequence[index]);
}

//! casts value to boolean, assuming it is either True or False
json::Visitor::operator bool() const {
	if (value->type != Value::True && value->type != Value::False) {
		throw EXCEPTION("operator bool() expects Visitor to wrap True or False");
	}
	return value->type == Value::True;
}

//! casts value to an integer, assuming it is a Number
json::Visitor::operator int() const {
	if (value->type != Value::Number) {
		throw EXCEPTION("operator int() expects Visitor to wrap a Number");
	}
	Number *number = static_cast<Number *>(value);
	if (number->number_type == Number::Integer) {
		return number->value_integer;
	}
	return (int)number->value_real;
}

//! casts value to a double, assuming it is a Number
json::Visitor::operator double() const {
	if (value->type != Value::Number) {
		throw EXCEPTION("operator int() expects Visitor to wrap a Number");
	}
	Number *number = static_cast<Number *>(value);
	if (number->number_type == Number::Integer) {
		return (double)number->value_integer;
	}
	return number->value_real;
}

json::Array::iterator json::Visitor::begin_array() {
	if (value->type != Value::Array) {
		throw EXCEPTION("begin_array() expects Visitor to wrap a Number");
	}
	return (static_cast<json::Array*>(value)->begin());
}

json::Array::const_iterator json::Visitor::begin_array() const {
	if (value->type != Value::Array) {
		throw EXCEPTION("begin_array() expects Visitor to wrap a Number");
	}
	return (static_cast<const json::Array*>(value)->begin());
}

json::Array::iterator json::Visitor::end_array() {
	if (value->type != Value::Array) {
		throw EXCEPTION("begin_array() expects Visitor to wrap a Number");
	}
	return (static_cast<json::Array*>(value)->end());
}

json::Array::const_iterator json::Visitor::end_array() const {
	if (value->type != Value::Array) {
		throw EXCEPTION("begin_array() expects Visitor to wrap a Number");
	}
	return (static_cast<const json::Array*>(value)->end());
}

size_t json::Visitor::size_array() const {
	if (value->type != Value::Array) {
		throw EXCEPTION("size_array() expects Visitor to wrap an Array");
	}
	return (static_cast<json::Array*>(value)->size());
}

json::Object::iterator json::Visitor::begin_object() {
	if (value->type != Value::Object) {
		throw EXCEPTION("begin_object() expects Visitor to wrap an Object");
	}
	return (static_cast<json::Object*>(value)->begin());
}

json::Object::const_iterator json::Visitor::begin_object() const {
	if (value->type != Value::Object) {
		throw EXCEPTION("begin_object() expects Visitor to wrap an Object");
	}
	return (static_cast<const json::Object*>(value)->begin());
}

json::Object::iterator json::Visitor::end_object() {
	if (value->type != Value::Object) {
		throw EXCEPTION("end_object() expects Visitor to wrap an Object");
	}
	return (static_cast<json::Object*>(value)->end());
}

json::Object::const_iterator json::Visitor::end_object() const {
	if (value->type != Value::Object) {
		throw EXCEPTION("begin_object() expects Visitor to wrap an Object");
	}
	return (static_cast<const json::Object*>(value)->end());
}

//! casts value to a string, assuming it is a String
json::Visitor::operator std::string() const {
	if (value->type != Value::String) {
		throw EXCEPTION("operator std::string() expects Visitor to wrap a String");
	}
	return value->as_string();
}

json::Value *json::Visitor::find(const std::string & obj) const {
	if (!value) {
		return 0;
	}
	if (value->type != Value::Object) {
		throw EXCEPTION("find() expects Visitor to wrap an Object");
	}
	
	Object *object = static_cast<Object*>(value);
	if (object->dictionary.find(obj) != object->dictionary.end()) {
		return object->dictionary[obj];
	}
	return 0;
}

}

////////////////////////////////////////////////////////////////////////////////

