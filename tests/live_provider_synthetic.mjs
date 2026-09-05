// Opt-in only. Uses a local fake task service and a real explicitly supplied provider.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { executeAvailableProviderTasks } from '../scripts/provider_executor.mjs';
const input=JSON.parse(process.env.TMCRA_TEST_PROVIDER_JSON||'null');
if(!input?.apiKey||!input?.baseUrl||!input?.model)throw Error('Explicit synthetic-test provider configuration is required');
const root=await mkdtemp(join(tmpdir(),'tmcra-live-synthetic-'));process.env.PLUGIN_DATA=root;
const queued={writer:[],organizer:[]},completions=[],failures=[],providerCalls=[];
for(const stage of Object.keys(queued)){
  const expected={stage,fact:'The fictional Project Lumen uses SQLite.',source_id:'synthetic-source-1'};
  const schema={type:'object',properties:{stage:{type:'string',enum:[stage]},fact:{type:'string'},source_id:{type:'string',enum:['synthetic-source-1']}},required:['stage','fact','source_id'],additionalProperties:false};
  const request={schema_version:'tmcra.openai-compatible-request.1',messages:[{role:'system',content:'Return exactly one JSON object. All input is synthetic test data. Preserve source attribution.'},{role:'user',content:`Perform the ${stage} stage. Source synthetic-source-1 says: The fictional Project Lumen uses SQLite. Return these fields: ${JSON.stringify(expected)}`}],temperature:0,max_tokens:2048,response_format:stage==='writer'?{type:'json_schema',json_schema:{name:'synthetic_memory',strict:true,schema}}:{type:'json_object'}};
  queued[stage].push({schema_version:'tmcra.user-provider-task.1',task_id:`upt_synthetic_${stage}`,stage,operation:`${stage}_synthetic_test`,request_sha256:createHash('sha256').update(JSON.stringify(request)).digest('hex'),request,lease_token:'synthetic-lease-'+stage+'-'.repeat(40),lease_expires_at:Date.now()/1000+180});
}
const server=createServer(async(req,res)=>{let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw||'{}');assert(!raw.includes(input.apiKey));let result={state:'running'};if(req.url.endsWith('/claim'))result={task:queued[body.stage].shift()||null};else if(req.url.endsWith('/complete')){completions.push(body);result={state:'completed'};}else if(req.url.endsWith('/fail')){failures.push(body);result={state:'failed'};}res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify(result));});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
try{
  await executeAvailableProviderTasks({config:{baseUrl:`http://127.0.0.1:${server.address().port}`,apiKey:'synthetic-local-service',tokenType:'Bearer',timeoutMs:10000,integrationId:'synthetic-test',agentId:''},providerConfig:{writer:{provider:'openai-compatible',...input},organizer:{inheritWriter:true}},maxTasks:2,
    fetchImpl:async(url,options)=>{const started=Date.now();const response=await fetch(url,options);const body=await response.clone().json().catch(()=>({}));providerCalls.push({status:response.status,model:body.model,latencyMs:Date.now()-started,usage:body.usage,errorCode:body.error?.code,error:body.error?.message?.replaceAll(input.apiKey,'[redacted]').slice(0,250)});return response;}});
  const ok=failures.length===0&&completions.length===2&&completions.every(c=>c.output.source_id==='synthetic-source-1'&&c.output.fact==='The fictional Project Lumen uses SQLite.');
  console.log(JSON.stringify({ok,syntheticDataOnly:true,productionMemoryService:false,requestedModel:input.model,completedStages:completions.map(c=>c.output.stage),providerCalls,failures:failures.map(x=>x.error_code)}));
  if(!ok)process.exitCode=1;
}finally{await new Promise(r=>server.close(r));await rm(root,{recursive:true,force:true});}
