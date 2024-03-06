from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

print(output_parser.parse("hi, bye"))
# >> ['hi', 'bye']
