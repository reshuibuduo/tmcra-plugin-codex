import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { controlKey, beginMemoryTurn, suppressMemoryTurn, mayWrite, memoryPolicy, recordMemoryActivity, setMemoryMode } from '../scripts/memory_controls.mjs';
import { createMemoryActions } from '../scripts/memory_center.mjs';

const root=await mkdtemp(join(tmpdir(),'tmcra-chat-confirm-'));process.env.TMCRA_MEMORY_STATE_DIR=join(root,'controls');
const posts=[];const http=createServer(async(req,res)=>{let body='';for await(const chunk of req)body+=chunk;if(req.method==='POST'&&req.url.endsWith('/feedback'))posts.push({path:req.url,body:JSON.parse(body),key:req.headers['idempotency-key']});res.writeHead(201,{'Content-Type':'application/json'});res.end(JSON.stringify({effective:true,correction_index_status:'pending'}));});
await new Promise(r=>http.listen(0,'127.0.0.1',r));const config={baseUrl:`http://127.0.0.1:${http.address().port}`,apiKey:'isolated-chat-test-key'};
await writeFile(join(root,'config.json'),JSON.stringify({...config,globalScope:'test-global',projectScopePrefix:'test-project'}));
const env={...process.env,TMCRA_CONFIG_FILE:join(root,'config.json'),TMCRA_BASE_URL:config.baseUrl,TMCRA_API_KEY:config.apiKey,PLUGIN_DATA:join(root,'plugin'),TMCRA_LOCAL_PROVIDER_CONFIG:join(root,'absent-provider.json')};
function client(capabilities,answer){
  const child=spawn(process.execPath,[resolve('scripts/mcp_server.mjs')],{env,cwd:process.cwd(),stdio:['pipe','pipe','pipe'],windowsHide:true});let seq=0;const pending=new Map();let asked=0;
  const send=value=>child.stdin.write(JSON.stringify({jsonrpc:'2.0',...value})+'\n');
  createInterface({input:child.stdout}).on('line',line=>{const msg=JSON.parse(line);if(msg.method==='elicitation/create'){asked++;void Promise.resolve(answer(msg)).then(result=>send({id:msg.id,result}));return;}const p=pending.get(msg.id);if(p){pending.delete(msg.id);clearTimeout(p.timer);msg.error?p.reject(Error(msg.error.message)):p.resolve(msg.result);}});
  const request=(method,params)=>new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>reject(Error('MCP test timed out')),8000);pending.set(id,{resolve,reject,timer});send({id,method,params});});
  return {child,request,get asked(){return asked;},async init(){await request('initialize',{protocolVersion:'2025-11-25',capabilities,clientInfo:{name:'confirmation-contract',version:'1'}});},async close(){child.stdin.end();child.kill();for(const p of pending.values())clearTimeout(p.timer);}};
}
let c;
try{
  const k=controlKey(config,'guard-test');const older=await beginMemoryTurn(k,'s','older');const vetoed=await beginMemoryTurn(k,'s','correction');await suppressMemoryTurn(k,'s');assert.equal(await mayWrite(older),true);assert.equal(await mayWrite(vetoed),false);const next=await beginMemoryTurn(k,'s','next');assert.equal(await mayWrite(next),true);assert.equal(await mayWrite(vetoed),false);
  await beginMemoryTurn(k,'parent','p-turn');const sub=await beginMemoryTurn(k,'parent:subagent:a','child-turn');await suppressMemoryTurn(k,'parent');assert.equal(await mayWrite(sub),false);
  for(const outcome of ['unavailable','decline','cancel','unchecked','accept']){
    const before=posts.length;c=client(outcome==='unavailable'?{}:{elicitation:{form:{}}},msg=>{assert.equal(posts.length,before,'no POST before human decision');assert.match(msg.params.message,/Old remembered fact/);assert.match(msg.params.message,/Corrected fact/);return {action:outcome==='unchecked'?'accept':outcome,content:{confirm:outcome==='accept'}};});await c.init();
    const args={session_id:'chat-session',project_path:root,project_id:'confirmation-test'};
    const dash=await c.request('tools/call',{name:'tmcra_memory_control',arguments:{...args,operation:'dashboard'}});assert.equal(dash.isError,false,JSON.stringify(dash));const scope=dash.structuredContent.scope;const key=controlKey(config,scope);
    const capture=await beginMemoryTurn(key,args.session_id,outcome);await recordMemoryActivity(capture,{kind:'recall',layers:[{scope,sources:[{memory_id:'source-a',content:'Old remembered fact'}]}]});
    const reply=await c.request('tools/call',{name:'tmcra_memory_control',arguments:{...args,operation:'feedback',action:'correct',memory_ids:['source-a'],replacement:'Corrected fact',idempotency_key:'correction-logical-one',confirmed:true}});
    assert.equal(reply.isError,false,JSON.stringify(reply));assert.equal(posts.length,before+(outcome==='accept'?1:0));assert.equal(c.asked,outcome==='unavailable'?0:1);assert.equal(await mayWrite(capture),false,'confirmation turn never backfills');if(outcome==='accept')assert.equal(reply.structuredContent.effective,true);else assert.equal(reply.structuredContent.applied,false);await c.close();c=null;
  }
  const key=controlKey(config,'boundary');await beginMemoryTurn(key,'s','b');let called=false;const invoke=createMemoryActions({config,scope:'boundary',globalScope:'g',sessionId:'s',confirmFeedback:async()=>{called=true;return 'accepted';},request:async()=>{throw Error('Unexpected remote request');}});
  await assert.rejects(invoke('feedback',{scope:'foreign',action:'correct',memory_ids:['s'],replacement:'new',idempotency_key:'boundary-key'}),/outside/);assert.equal(called,false);
  const capture=await beginMemoryTurn(key,'s','context');await recordMemoryActivity(capture,{kind:'recall',layers:[{scope:'boundary',sources:[{memory_id:'source-a',content:'old'}]}]});
  const changed=createMemoryActions({config,scope:'boundary',sessionId:'s',confirmFeedback:async()=>{await setMemoryMode(key,'s','off');return 'accepted';},request:async()=>{throw Error('No POST after context changes');}});
  assert.equal((await changed('feedback',{action:'correct',memory_ids:['source-a'],replacement:'new',idempotency_key:'context-key'})).status,'context_changed');
  console.log(JSON.stringify({ok:true,mcpElicitation:true,noPostBeforeConsent:true,declineCancelUnsupported:true,modelBooleanCannotApprove:true,turnSuppression:true,olderQueuePreserved:true,scopeAndContextBound:true}));
}finally{await c?.close();await new Promise(r=>http.close(r));await rm(root,{recursive:true,force:true});}
