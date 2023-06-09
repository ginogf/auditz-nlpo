import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';
const fs = require('fs');
const path = require('path');

// Function to read contract files and convert them to embeddings
async function convertContractsToEmbeddings(directory) {
  const files = fs.readdirSync(directory);
  const embeddings = [];

  for (const file of files) {
    const filePath = path.join(directory, file);
    const contractCode = fs.readFileSync(filePath, 'utf-8');

    // Split contract code into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    const chunks = await textSplitter.createDocuments([contractCode]);

    // Convert each chunk of contract code to embedding
    const embeddingsArrays = await new OpenAIEmbeddings().embedDocuments(
      chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
    );

    embeddings.push(...embeddingsArrays);
  }

  return embeddings;
}

// Run the function
convertContractsToEmbeddings('./contracts')
  .then(embeddings => console.log(embeddings))
  .catch(err => console.error(err));
