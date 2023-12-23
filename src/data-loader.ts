import fs from 'fs';
import readline from 'readline';

export const loadLJSONLines = async <L>(filePath: string, lineNumbers: number[]) => {
	const rl = readline.createInterface({
		input: fs.createReadStream(filePath),
		crlfDelay: Infinity
	});
	let lineNumber: number = 0;
	const lines: L[] = [];
	for await (const line of rl) {
		if (lineNumbers.includes(lineNumber)) lines.push(JSON.parse(line));
		lineNumber++;
	};
	return lines;
};